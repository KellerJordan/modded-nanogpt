# Contra Muon results

Contra Muon (https://github.com/nilin/contra-muon) modifies the Muon Newton-Schulz momentum update by subtracting `CONTRA_MUON / 2` times the operator-normalized momentum gradient

Everything else follows PR 274 NorMuon-lite with row/column variance normalization and u/w-floor postprocessing.

This run set uses `CONTRA_MUON=0.4`, `lr=0.0375`, `u/w-floor=0.35`, and
terminates at 3225 steps. Across 16 non-cherry-picked seeds, the step 3225
mean validation loss is 3.27854062. Under the Track 3 README's one-sided
z-test with `sigma=0.0016`, this gives `p=0.000131920034357`.
