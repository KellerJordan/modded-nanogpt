# SOAP record October 9 2024

* New sample efficiency record: <3.28 validation loss in 3.15B tokens
* Uses SOAP optimizer ([Vyas et al. 2024](https://arxiv.org/abs/2409.11321))
* 363ms/step - not a new wallclock record (SOAP is in active development to reduce the wallclock overhead for distributed training, so this may change)
* Set by Nikhil Vyas @vyasnikhil96. Hyperparameters also tuned slightly by me
* [https://x.com/vyasnikhil96/status/1842656792217858063](https://x.com/vyasnikhil96/status/1842656792217858063)
* [https://github.com/nikhilvyas/modded-nanogpt-SOAP/tree/master](https://github.com/nikhilvyas/modded-nanogpt-SOAP/tree/master)

