# Forward and timing audit

Frozen candidate commit: c5be2fbf86c918e91b9e283b4044595bb74a350f.

Compared approx_backward/reference.py with the exact PR360 FA3 package's public FlashAttnVarlenFunc.forward and private _flash_attn_forward definitions. The q/k/v, cumulative lengths, maximum lengths, softmax scale, causal flag and window arguments agree. Optional qv/descales/used lengths are None; attention_chunk/softcap/sm_margin are zero; num_splits is 1; pack_gqa is None. Both paths share the same contiguity handling in _flash_attn_forward. The custom autograd wrapper changes backward, not attention forward.

The training timer starts before prefix-table construction. backend.after_backward runs after each backward and before its optimizer step, inside the timed training loop. Snapshot copies execute within the backward graph. Calibration uses complete-document snapshots from current training data, compares full and restricted gradients, takes the maximum across ranks, copies the selected windows and synchronizes before recording duration. The original clock stops only after final ships and validation-row preparation. Full validation and final diagnostic serialization occur after this clock stop. The schedule keeps 1194 training updates and original data.

The first two real eight-H100 candidate runs passed all 11 graph configurations, including the duplicate untimed replay for the one-step configuration at591. No numerical-check or compilation time is represented as training speedup. Candidate source, snapshot data and calibration state are not transferred between runs.
