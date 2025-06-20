import torch
import torch.nn as nn
from typing import Tuple, Optional

"""
Standardized MLM masking approach for consistency
"""

class ProteinMasker(nn.Module):
    def __init__(self, tokenizer, mask_rate=0.15):
        """
        Implements the masking scheme from DSM with a default 15% mask probability.
        """
        super().__init__()
        self.mask_token_id = tokenizer.mask_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.mask_rate = mask_rate

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: The input token IDs.
            attention_mask: Optional attention mask.
            
        Returns:
            Tuple of (masked_input_ids, labels)
        """
        eps = 1e-3
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        # Default to 15% masking if t not provided
        t = torch.full((batch_size,), self.mask_rate, device=device)

        
        p_mask = t[:, None].repeat(1, seq_len)
        mask_indices = torch.rand(batch_size, seq_len, device=device) < p_mask
        
        # Prevent cls and eos from being masked
        cls_mask = input_ids == self.cls_token_id
        eos_mask = input_ids == self.eos_token_id
        mask_indices = mask_indices & ~cls_mask & ~eos_mask & attention_mask.bool()
        
        # Ensure at least one token is masked per sequence
        for i in range(batch_size):
            if not mask_indices[i].any() and attention_mask[i].sum() > 2:  # More than just CLS/EOS
                # Find valid positions (not CLS/EOS and has attention)
                valid_positions = (~cls_mask[i]) & (~eos_mask[i]) & attention_mask[i].bool()
                if valid_positions.any():
                    # Get indices of valid positions
                    valid_indices = valid_positions.nonzero(as_tuple=True)[0]
                    # Randomly select one position to mask
                    random_idx = valid_indices[torch.randint(0, valid_indices.size(0), (1,), device=device)]
                    mask_indices[i, random_idx] = True
        
        # Create masked input
        masked_input_ids = torch.where(mask_indices, self.mask_token_id, input_ids)
        
        # Create labels for loss computation
        labels = input_ids.clone()
            
        non_mask_indices = ~mask_indices | (attention_mask == 0)
        labels[non_mask_indices] = -100
        
        return masked_input_ids, labels


if __name__ == "__main__":
    from transformers import EsmTokenizer
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    test_seqs = [
        'MNFKYKLYSYITIFQIILILPTIVASNERCIALGGVCKDFSDCTGNYKPIDKHCDGSNNIKCCIRKIECPTSQNSNFTISGKNKEDEALPFIFKSEGGCQNDKNDNGNKINGKIGYTCAGITPMVGWKNKENYFSYAIKECTNDTNFTYCAYKLNENKFREGAKNIYIDKYAVAGKCNNLPQPAYYVCFDTSVNHGSGWSSKTITANPIGNMDGREYGLLLNKKSREKYINIVKNDSSQEKYLNGWLSRADDREKYCNNYCTSNCNCDNSASKASVSSNTNTTDIYNSVNTVDSDICNCDDNEPTDFLDDDYINNEEEIDEEIIDQEEY',
        'MYRTALYFTVCSIWLCQIITGVLSLKCKCDLCKDKNYTCITDGYCYTSATLKDGVILYNYRCLDLNFPMRNPMFCHKQIPIHHEFTLECCNDRDFCNIRLVPKLTPKDNATSDTSLGTIEIAVVIILPTLVICIIAMAIYLYYQNKRSTHHHLGLGDDSIEAPDHPILNGVSLKHMIEMTTSGSGSGLPLLVQRSIARQIQLVEIIGQGRYGEVWRGRWRGENVAVKIFSSREERSWFREAEIYQTVMLRHDNILGFIAADNKGVLSLKCKCDLCKDKNYTCITDGYCYTSATLKDGVILYNYRQLGASLNRFXVYALGLIFWEISRRCNVGGIYDEYQLPFYDAVPSDPTIEEMRRVVCVERQRPSIPNRWQSCEALHVMSKLMKECWYHNATARLTALRIKKTLANFRASEELKM'
    ]
    tokenized = tokenizer(test_seqs, return_tensors="pt", padding=True)
    test_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    
    masker = ProteinMasker(tokenizer)
    
    # Test with default 15% masking
    masked_ids1, labels1 = masker.forward(test_ids.clone(), attention_mask)
    
    # Test with custom probability
    custom_t = torch.tensor([0.2, 0.3])
    masked_ids2, labels2 = masker.forward(test_ids.clone(), attention_mask, custom_t)
    
    print("Original:")
    print(f'Start: {test_ids[0][:20].tolist()}')
    print(f'End:   {test_ids[0][-20:].tolist()}')
    print(f'Start: {test_ids[1][:20].tolist()}')
    print(f'End:   {test_ids[1][-20:].tolist()}')
    print("Masked (15%):")
    print(f'Start: {masked_ids1[0][:20].tolist()}')
    print(f'End:   {masked_ids1[0][-20:].tolist()}')
    print(f'Start: {masked_ids1[1][:20].tolist()}')
    print(f'End:   {masked_ids1[1][-20:].tolist()}')
    print("Masked (custom):")
    print(f'Start: {masked_ids2[0][:20].tolist()}')
    print(f'End:   {masked_ids2[0][-20:].tolist()}')
    print(f'Start: {masked_ids2[1][:20].tolist()}')
    print(f'End:   {masked_ids2[1][-20:].tolist()}')
    
    # Test with seed
    torch.manual_seed(42)
    masked_ids4, labels4 = masker.forward(test_ids.clone())
    torch.manual_seed(42)
    masked_ids5, labels5 = masker.forward(test_ids.clone())
    
    print("\nAfter setting seed:")
    print("Are they equal?", torch.equal(masked_ids4, masked_ids5))
