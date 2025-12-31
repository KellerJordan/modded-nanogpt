## New WR 149.6s: MuonCustomSizing, perform mlp and attn reduce scatter in shared call

This PR builds on all recent WR improvements including PR #131. Updates:
* Add Muon Custom Sizing

```
class Muon(torch.optim.Optimizer):
    """

    ...

    Custom distributed sizing:
    The model stores all attn and mlp weights in the same shape, and then updates the view as 
    needed on the forward pass. This enables attn and mlp weights to be contained within the same 
    dist.reduce_scatter_tensor() call. The model architecture has been customized to enable 
    (n_attn_layers+n_mlp_layers*2)%4==0 for batching across 8 GPUs with zero padding. The scheduling is:
        1. reduce scatter smear_gate (1 param 7 padding params)
        2. reduce scatter attn_gate (10 params 6 padding params)
        3. reduce scatter attn/mlp round 1 (10 attn params 6 mlp params)
        4. reduce scatter attn/mlp round 2 (16 mlp params)
        5. wait on step 1, then compute NS of 1 and schedule all gather
        6. wait on step 2, then compute NS of 2 and schedule all gather
        7. wait on step 3, then compute NS of 3 and schedule all gather
            GPUs receive [2 ATTN, 2 ATTN, 2 ATTN, 2 ATTN, 2 ATTN, 2 MLP, 2 MLP, 2 MLP]
            GPUs that receive params of type attn reshape before NS
        8. wait on 4, then compute NS of 4 and schedule all gather
        9. wait for each all gather to complete and update params
    Empirically, leading with small params provides an additional 0.2s improvement.
    """

    def generate_custom_param_groups(self, params):
        # implementation requires that a single GPU does not recieve both attn 
        # and mlp params when a param group is split across GPUs
        module_ranks = {
            'smear_gate': 1, # 1 param
            'attn_gate': 2, # 10 params
            'attn': 3, # 10 params
            'mlp': 4, # 22 params
        }
        params = list(params)
        params.sort(key=lambda x: module_ranks.get(x.module))
        idx = 0
        group_sizes = [1,10,16,16]
        assert len(params)==sum(group_sizes)
        param_groups = []
        for size in group_sizes:
            group_params = params[idx:idx+size]
            param_groups.append(dict(params=group_params))
            idx += size
        return param_groups

    if getattr(params[module_idx],'module','none')=='attn':
        batch = 4 * original_shape[0]
        d1 = original_shape[1] 
        d2 = original_shape[2] // 4
        batched = batched_update_grads.view(batch, d1, d2)
        v_chunk = newton_schulz_triton(batched)
        v_chunk = v_chunk.view(original_shape)
```

Reshaping attn on forward pass:
```
self.qkvo_w = nn.Parameter(torch.empty(self.hdim, self.dim*4))
q, k, v = F.linear(x, self.qkvo_w.view(4,self.hdim, self.dim)[:3].flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
y = F.linear(y, self.qkvo_w.view(4,self.hdim, self.dim)[3].type_as(y))
```

Skipping ML validation since ML is identical.

Rerunning prior record: 150.3843 [150.393,150.347,150.413]

New runtime: 149.6905 [149.686,149.678,149.775,149.623]