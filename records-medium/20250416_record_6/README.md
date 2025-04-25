1. remove FP8 lm head, 6710→6450 steps, ~0 wall clock change
2. logits softcap tanh→ISRU, ~1.5ms/step faster, slightly better (2.91970 -> 2.91944)
3. New sharded mixed precision Muon, remove CastedLinear, ~3.6ms/step faster
4. merge qkv&o weight, ~0.8ms/step faster
5. merge scalars weight, ~0.6ms/step faster
