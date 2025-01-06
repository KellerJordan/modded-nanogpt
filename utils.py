import torch
import torch.nn as nn
from typing import Tuple

"""
Standardized MLM masking approach for consistency
"""

class ProteinMasker(nn.Module):
    def __init__(self, tokenizer):
        """
        Baseline: 80% replaced with [MASK], 10% replaced with a random token, and 10% unchanged.
        """
        super().__init__()
        self.mask_token_id = tokenizer.mask_token_id
        standard_tokens = [tokenizer.convert_tokens_to_ids(tok) for tok in tokenizer.all_tokens if tok not in tokenizer.all_special_tokens]
        canonical_amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        canonical_amino_acids_ids = tokenizer.convert_tokens_to_ids(list(canonical_amino_acids))
        low_range = min(canonical_amino_acids_ids)
        high_range = max(canonical_amino_acids_ids)
        self.register_buffer("standard_tokens", torch.tensor(standard_tokens, dtype=torch.int32))
        self.register_buffer("special_tokens", torch.tensor(tokenizer.all_special_ids, dtype=torch.int32))
        self.register_buffer("low_range", torch.tensor(low_range, dtype=torch.int32))
        self.register_buffer("high_range", torch.tensor(high_range, dtype=torch.int32))

    def __call__(
            self, input_ids: torch.Tensor, mask_prob: torch.Tensor, keep_replace_prob: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = input_ids.clone()

        # Create special tokens mask using broadcasting
        special_tokens_mask = (input_ids[..., None] == self.special_tokens).any(-1)

        mlm_prob = mask_prob + keep_replace_prob * 2
        mask_portion = mask_prob / mlm_prob

        # Create probability matrix and mask special tokens
        probability_matrix = torch.ones_like(labels, dtype=torch.float) * mlm_prob
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Create masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # mask_prob% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full_like(probability_matrix, mask_portion)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id

        # keep_replace_prob% of the time, we replace masked input tokens with random word
        replacement_idxs = torch.bernoulli(
            torch.full_like(probability_matrix, 0.5)
        ).bool() & masked_indices & ~indices_replaced
        random_token_idxs = torch.randint(
            self.low_range, self.high_range, (replacement_idxs.sum(),),
            dtype=input_ids.dtype, device=replacement_idxs.device
        )
        input_ids[replacement_idxs] = self.standard_tokens[random_token_idxs]

        # The rest of the time (keep_replace_prob% of the time again) we keep the masked input tokens unchanged
        return input_ids, labels


if __name__ == "__main__":
    from transformers import EsmTokenizer
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    test_seqs = [
        'MNFKYKLYSYITIFQIILILPTIVASNERCIALGGVCKDFSDCTGNYKPIDKHCDGSNNIKCCIRKIECPTSQNSNFTISGKNKEDEALPFIFKSEGGCQNDKNDNGNKINGKIGYTCAGITPMVGWKNKENYFSYAIKECTNDTNFTYCAYKLNENKFREGAKNIYIDKYAVAGKCNNLPQPAYYVCFDTSVNHGSGWSSKTITANPIGNMDGREYGLLLNKKSREKYINIVKNDSSQEKYLNGWLSRADDREKYCNNYCTSNCNCDNSASKASVSSNTNTTDIYNSVNTVDSDICNCDDNEPTDFLDDDYINNEEEIDEEIIDQEEY',
        'MYRTALYFTVCSIWLCQIITGVLSLKCKCDLCKDKNYTCITDGYCYTSATLKDGVILYNYRCLDLNFPMRNPMFCHKQIPIHHEFTLECCNDRDFCNIRLVPKLTPKDNATSDTSLGTIEIAVVIILPTLVICIIAMAIYLYYQNKRSTHHHLGLGDDSIEAPDHPILNGVSLKHMIEMTTSGSGSGLPLLVQRSIARQIQLVEIIGQGRYGEVWRGRWRGENVAVKIFSSREERSWFREAEIYQTVMLRHDNILGFIAADNKGVLSLKCKCDLCKDKNYTCITDGYCYTSATLKDGVILYNYRQLGASLNRFXVYALGLIFWEISRRCNVGGIYDEYQLPFYDAVPSDPTIEEMRRVVCVERQRPSIPNRWQSCEALHVMSKLMKECWYHNATARLTALRIKKTLANFRASEELKM'
    ]
    test_ids = tokenizer(test_seqs, return_tensors="pt", padding=True).input_ids.to(torch.int32)
    masker = ProteinMasker(tokenizer)
    print(masker.mask_token_id)
    print(masker.special_tokens)

    # First set of masking with different probabilities
    mask_prob = torch.tensor(0.4)
    keep_replace_prob = torch.tensor(0.1)
    masked_ids1, labels1 = masker(test_ids.clone(), mask_prob, keep_replace_prob)
    masked_ids2, labels2 = masker(test_ids.clone(), mask_prob, keep_replace_prob)

    print("Before setting seed:")
    print("Original: ", test_ids[0][:20].tolist())
    print("Masking 1:", masked_ids1[0][:20].tolist()) 
    print("Masking 2:", masked_ids2[0][:20].tolist())
    print("Are they equal?", torch.equal(masked_ids1, masked_ids2))

    # Now with seed
    torch.manual_seed(42)
    masked_ids3, labels3 = masker(test_ids.clone(), mask_prob, keep_replace_prob)
    torch.manual_seed(42)
    masked_ids4, labels4 = masker(test_ids.clone(), mask_prob, keep_replace_prob)

    print("\nAfter setting seed:")
    print("Original: ", test_ids[0][:20].tolist())
    print("Masking 3:", masked_ids3[0][:20].tolist())
    print("Masking 4:", masked_ids4[0][:20].tolist()) 
    print("Are they equal?", torch.equal(masked_ids3, masked_ids4))
