import torch
from typing import Tuple

"""
Standardized MLM masking approach for consistency
"""

class ProteinMasker(torch.nn.Module):
    def __init__(self, tokenizer, mlm_probability=0.15):
        """
        Initialize the ProteinMasker with the given tokenizer and masking parameters.
        Of the masked tokens, 80% are replaced with [MASK], 10% are replaced with a random amino acid token, and 10% are unchanged.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id = tokenizer.mask_token_id
        self.special_tokens = tokenizer.all_special_ids
        canonical_amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        canonical_amino_acids_ids = tokenizer.convert_tokens_to_ids(list(canonical_amino_acids))
        self.low_range = min(canonical_amino_acids_ids)
        self.high_range = max(canonical_amino_acids_ids)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # modified from
        # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/data/data_collator.py#L828
        labels = input_ids.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(low=self.low_range, high=self.high_range, size=labels.shape, dtype=torch.long, device=labels.device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels


if __name__ == "__main__":
    from transformers import EsmTokenizer
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    test_seqs = [
        'MNFKYKLYSYITIFQIILILPTIVASNERCIALGGVCKDFSDCTGNYKPIDKHCDGSNNIKCCIRKIECPTSQNSNFTISGKNKEDEALPFIFKSEGGCQNDKNDNGNKINGKIGYTCAGITPMVGWKNKENYFSYAIKECTNDTNFTYCAYKLNENKFREGAKNIYIDKYAVAGKCNNLPQPAYYVCFDTSVNHGSGWSSKTITANPIGNMDGREYGLLLNKKSREKYINIVKNDSSQEKYLNGWLSRADDREKYCNNYCTSNCNCDNSASKASVSSNTNTTDIYNSVNTVDSDICNCDDNEPTDFLDDDYINNEEEIDEEIIDQEEY',
        'MYRTALYFTVCSIWLCQIITGVLSLKCKCDLCKDKNYTCITDGYCYTSATLKDGVILYNYRCLDLNFPMRNPMFCHKQIPIHHEFTLECCNDRDFCNIRLVPKLTPKDNATSDTSLGTIEIAVVIILPTLVICIIAMAIYLYYQNKRSTHHHLGLGDDSIEAPDHPILNGVSLKHMIEMTTSGSGSGLPLLVQRSIARQIQLVEIIGQGRYGEVWRGRWRGENVAVKIFSSREERSWFREAEIYQTVMLRHDNILGFIAADNKGVLSLKCKCDLCKDKNYTCITDGYCYTSATLKDGVILYNYRQLGASLNRFXVYALGLIFWEISRRCNVGGIYDEYQLPFYDAVPSDPTIEEMRRVVCVERQRPSIPNRWQSCEALHVMSKLMKECWYHNATARLTALRIKKTLANFRASEELKM'
    ]
    test_ids = tokenizer(test_seqs, return_tensors="pt", padding=True).input_ids
    masker = ProteinMasker(tokenizer, mlm_probability=0.5)
    print(masker.mask_token_id)
    print(masker.special_tokens)
    print(masker.low_range, masker.high_range)

    # First set of masking
    masked_ids1, labels1 = masker(test_ids.clone())
    masked_ids2, labels2 = masker(test_ids.clone())

    print("Before setting seed:")
    print("Original: ", test_ids[0][:20].tolist())
    print("Masking 1:", masked_ids1[0][:20].tolist()) 
    print("Masking 2:", masked_ids2[0][:20].tolist())
    print("Are they equal?", torch.equal(masked_ids1, masked_ids2))

    # Now with seed
    torch.manual_seed(42)
    masked_ids3, labels3 = masker(test_ids.clone())
    torch.manual_seed(42)
    masked_ids4, labels4 = masker(test_ids.clone())

    print("\nAfter setting seed:")
    print("Original: ", test_ids[0][:20].tolist())
    print("Masking 3:", masked_ids3[0][:20].tolist())
    print("Masking 4:", masked_ids4[0][:20].tolist()) 
    print("Are they equal?", torch.equal(masked_ids3, masked_ids4))
