# New record 11/03/24


New NanoGPT training speed record: 3.28 FineWeb val loss in 10.8 minutes on 8xH100

Previous record: 12.0 minutes
Changelog:
- untied embed and head weights
- added RMSNorm after embed
- init head to zero

Driven by @Grad62304977

---

Technically, this is somewhat of an "any%" record, since untying the embedding and lm_head adds 39M parameters.

However, it doesn't change the number of active parameters or the inference throughput. Future records will stay constrained to 124M active parameters.

---

Like the last architectural change, this record was driven by @Grad62304977. I just finetuned some things and did bookkeeping.

---

Shoutout to @cloneofsimo whose scaling guide already suggests initializing the head to zero. This works quite well and is a significant fraction of the record.

