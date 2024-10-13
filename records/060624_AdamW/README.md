This is the log for my baseline AdamW training to which I compared the new Muon and SOAP optimizers.

just the log, which is in the old llm.c format ("tel" lines are val loss)

this was batch size 2^19, so ~5B tokens

was learning rate 0.0018, warmup=250, warmdown=2000, betas=(0.9, 0.95) IIRC

