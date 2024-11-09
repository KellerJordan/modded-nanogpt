This is a replication attempt for the record attempt described [here](https://x.com/leloykun/status/1854557419768254915) by @leloykun.

The original record could not be directly accepted because it showed a slower wallclock time than the previous record -
however, this was plausibly due to hardware differences, as the competitor's hardware was slightly slower.

Therefore, to certify this attempt as the new record, here I replicated it on my own hardware.
This did successfully reduce the wallclock time compared to the 11/07/24 record by ~11 seconds, however it also
resulted in an invalid val loss of 3.2824, above the threshold of 3.28.

The [original record attempt's reproducible log](https://github.com/leloykun/modded-nanogpt/blob/224f10d190677d9dc3c9c45da280078196a6fe40/records/110724_EmbeddingBetasCooldown/6c9d875b-ad91-46c9-9ede-2c7f998b9b16.txt) attained a val loss of 3.2798, just barely below the 3.28 threshold. So this difference is plausibly due to random inter-run variance.

This indicates that the true average val loss of the run may be worse than 3.28, meaning I am **unable to certify it as the new record.**

Ideally, all records should attain a low enough val loss such that >95% of runs attain below 3.28. Good evidence for this would be a single run
attaining <= 3.278. Previous records have adhered to this rule, but admittedly it's hard to define precisely and is therefore mostly a matter of taste.

