# Execution notes

The experiment plan was frozen while a Modal launch was pending. Before any
run outcome was observed, the operator requested that execution move to
Nebius. The same requested accelerator class and paired design were retained:
one dedicated node with 8 NVIDIA H100 80GB HBM3 GPUs, and both arms ran in the
same environment and TorchInductor cache.

## Stage 1 engineering screen

The Nebius ABBA screen completed without infrastructure failures. Every run
started with no GPU compute processes, and both candidate runs reported the
terminal TailEMA assertion.

| Order | Arm | Final loss | Train time |
|---:|---|---:|---:|
| 1 | baseline | 3.2773 | 76.172s |
| 2 | candidate | 3.2764 | 75.641s |
| 3 | candidate | 3.2774 | 75.686s |
| 4 | baseline | 3.2762 | 76.364s |

Mean baseline time was 76.2680s and mean candidate time was 75.6635s. The
screening ratio was 0.9920740, or 0.6045s / 0.7926% faster. Per the frozen
decision rule, this advanced to the ten-run accuracy confirmation.

Stage 1 is exploratory and is not claimed as confirmatory record evidence.

## Confirmatory order freeze

Before the first Stage 2 outcome returned, the Stage 3 order was fixed as
`ABBA, BAAB, ABBA, BAAB, ABBA, BAAB, ABBA, BAAB`, where A is baseline and B
is candidate. The runner SHA-256 was
`39edfec67cb70545d03935499b0fd0b41313646e3c2069bb4bcead67888652b1`.
The already-launched ten-run accuracy runner SHA-256 was
`dd81e1e1acbff03143d5bf53435992380d76d6b56e8fba8fe16eabc101f078a1`.

## Disclosed Stage 2 infrastructure failure

The first launch of candidate outcome 7 completed timed step 1288 at
75.587s, then stalled before producing its terminal validation value. Four
GPU ranks spun while four waited; there was no Xid, ECC, OOM, or final loss.
After 120 seconds without log progress, the torchrun parent received SIGTERM.
This launch is classified as a pre-outcome infrastructure failure and is not
an accuracy observation. Its console and hardware logs are retained.

After all GPU contexts released, two separate 8-rank NCCL all-reduce health
checks passed with the expected sum of 36 and zero reported ECC errors. The
retry runner starts at outcome 7, refuses to overwrite existing outcomes, and
adds a fixed ten-minute infrastructure timeout. Its SHA-256 is
`2b3382b81cc40d325b0cbd5a61669f03b9a4dd301e5d2bffd173d6db3520e339`.

The same fixed ten-minute per-run infrastructure timeout was then added to
the not-yet-launched Stage 3 runner. The block order and all statistical
decisions remained unchanged. The amended runner SHA-256 is
`f7792f58c2e77aed61dfb09509bbf8cdfd3a68701c7b2c6b8fad2a6d83b01dcb`.

Candidate outcome 9's first launch reproduced the same symptom before
producing a loss. Rank stack capture identified the cause more precisely:
some Python 3.12 forked Inductor workers deadlocked in a
`ProcessPoolExecutor` weakref lock while first-importing
`torch._inductor.runtime.triton_heuristics`; completed ranks then blocked
while materializing the GPU loss tensor. The eight rank stacks are retained.

For subsequent launches, a wrapper preloads that existing module in every
rank before Inductor forks. It sets `sys.argv[0]` back to `train_gpt.py`, so
each full log still embeds the exact training source. The wrapper changes no
Inductor/compile flags, performs no random draws, and runs before the benchmark
timer. Its SHA-256 is
`339cd25a2714fc8a7e0df11a3c2a3964715c3e460c3c9a3e9325720f5442d8e6`; the
outcome-9 resume runner SHA-256 is
`23f25dda1769f67a8e2829c870b5ee7eb97eb864b506a4311ea2428091f076be`.

The first wrapper invocation failed before compilation or training because
`runpy` did not place the training directory on `sys.path`; it could not
import `triton_kernels`. This setup failure is also retained separately. The
corrected wrapper resolves the training path and prepends its parent directory.
An eight-rank smoke test produced eight `PRELOAD_PATH_SMOKE_PASS` lines before
the next launch. Corrected wrapper SHA-256:
`0d80e06cf69c513e42c9f98d94c5111bb956f988dd6663211b780ed09ca212f8`.

Outcome 9 then completed with loss 3.2782. Outcome 10's first launch completed
timed step 1288 at 75.536s but again stalled before producing a loss. A second
stack capture showed why the rank-level preload was not sufficient: Inductor
launches a fresh Python coordinator, and that coordinator forks its own pool
of 32 workers. Those workers could still first-import `triton_heuristics` and
deadlock in the same Python 3.12 weakref callback. This launch has no loss, is
classified as a pre-outcome infrastructure failure, and is retained as
`10-candidate-attempt1.*`.

The next wrapper adds a narrowly scoped `sitecustomize` hook to the child
`PYTHONPATH`. It imports `triton_heuristics` only when the child command is
Inductor's `--kind=fork` compile-worker coordinator, before that coordinator
creates its worker pool. It does not change worker count, CPU affinity, I/O
priority, Inductor settings, RNG state, or timed training. A coordinator-path
smoke test passed before relaunch. Wrapper SHA-256:
`afd7de054fee7027d36e33c67ec2df4112559c6041eabaf644c3de432708b701`;
hook SHA-256:
`c92de5731bf52a6ff179f21eadf674f7f02e62b1d6b081c1f38b7ce521ff3a4c`.

## Stage 2 accuracy confirmation

The ten prespecified, loss-bearing candidate outcomes were:

| Outcome | Final loss | Train time |
|---:|---:|---:|
| 1 | 3.2812 | 75.626s |
| 2 | 3.2794 | 75.694s |
| 3 | 3.2758 | 75.654s |
| 4 | 3.2755 | 75.726s |
| 5 | 3.2778 | 75.662s |
| 6 | 3.2788 | 75.655s |
| 7 | 3.2791 | 75.592s |
| 8 | 3.2774 | 75.635s |
| 9 | 3.2782 | 75.487s |
| 10 | 3.2783 | 75.566s |

Mean loss was 3.27815000 with sample standard deviation 0.00168143. The
prespecified one-sided one-sample t-test against 3.28 returned
`p=0.0034731121`, so the accuracy gate passed. Mean candidate runtime was
75.6297s with sample standard deviation 0.0680s.

After this gate passed, the candidate's default scheduled iterations were
changed from 1285 to the already-tested effective value 1273; the extension
remains 15, for 1288 total steps. Stage 2 used the same value through the
`NUM_SCHEDULED_ITERATIONS=1273` environment control. The final candidate
source SHA-256 is
`ee9586a5579a94d135ad77a826538553566ecd398b9ca3c49729ca3107969063`.
Stage 3 will therefore provide both timing and final-source accuracy evidence.

The not-yet-launched Stage 3 runner now uses the same neutral preload for both
arms. Its frozen order and statistical decisions are unchanged. Final runner
SHA-256: `73e3cac18e07bf18423be46c1e9d8d10e203d51f4d3e1d86d7c508a4ea0944bb`.

## Disclosed Stage 3 infrastructure rejection

The first launch of block 1 completed two loss-bearing outcomes, then its
third launch completed timed step 1288 at 75.515s and stalled before producing
a loss. Per the frozen protocol, the whole block was rejected; its two
completed results, failed launch, driver log, hardware records, and stack
capture are retained under `rejected-block01-attempt1/` and are excluded from
confirmatory statistics.

The capture exposed one remaining path error. Inductor's
`python_subprocess_env()` deliberately reconstructs child `PYTHONPATH` from
the rank's `sys.path`, overwriting the wrapper's environment-only addition.
The live coordinator therefore lacked the hook directory. The wrapper was
corrected to prepend the hook to both `sys.path` and `PYTHONPATH`; no compiler
or training setting changed. Corrected wrapper SHA-256:
`a1e7d3b649494b7ce7483e532d1581aace2090d8b344c07ec6fc96937c85bb50`.

Before restarting Stage 3, the actual live coordinator environment was
checked and contained `/home/karan/speedrun/bin/inductor-preload`. A complete
eight-GPU candidate stress run then reached terminal validation with loss
3.2811 in 75.471s and exited zero. Confirmatory block numbering restarted at
block 1 only after all GPUs were idle and the rejected files were archived.

The restarted block 1 completed and was accepted. Its baseline times were
76.312s and 76.384s; candidate times were 75.646s and 75.639s. The block
geometric runtime ratio was approximately 0.99076.

Block 2's first attempt then completed its leading candidate outcome, but the
second (baseline) launch completed timed step 1300 at 76.190s and stalled
before producing a loss. The whole block was rejected and retained under
`rejected-block02-attempt1/`. New stacks showed the weakref callback could run
later inside an already-imported `triton_heuristics` function, so import
preloading alone could not eliminate the Python 3.12 fork race.

Spawn-based Inductor workers were explored only in separate infrastructure
stress runs. Spawn avoided the inherited-lock failure but was not used for any
record outcome: it is an extra Inductor setting prohibited by the Track 1
rules, and the 8-by-32-worker spawn stress was itself unstable.

The final infrastructure-only fix keeps Inductor's default subprocess/fork
mode and default 32 compiler workers. A coordinator `sitecustomize` hook
registers `gc.disable` as an `after_in_child` callback. Cyclic GC is therefore
disabled only in the short-lived forked compiler workers, preventing them from
collecting the inherited executor whose shutdown lock may be held by a vanished
coordinator thread. Reference counting, compilation settings, model code, RNG,
and timed training are unchanged. A direct fork-child test passed, followed by
two complete consecutive candidate stress runs: 75.585s/loss 3.2780 and
75.629s/loss 3.2777, both with clean exits. Final wrapper SHA-256:
`cf40060aef9b8b44cde3df493dec448589cda274f0fc42dfde59d6b79296a164`;
final hook SHA-256:
`265433d1f55e5606f98b8603b0101533605c505397877a38f38000403fcc8aff`.

The runner gained only a validated `START_BLOCK` resume control so accepted
block 1 remained untouched while the archived block 2 was relaunched. Order,
outcome checks, timeout, and statistics were unchanged. Resume runner SHA-256:
`c3317ebd649888a96c5e5706e4d2d53dffaba13f2cdeb352b50a2d1e8037b484`.

## Stage 3 timing confirmation

The resumed confirmation completed all eight prespecified blocks with 16
loss-bearing outcomes per arm. No block was rejected after the final restart.

| Block | Order | Baseline mean | Candidate mean | Ratio |
|---:|:---:|---:|---:|---:|
| 1 | ABBA | 76.3480s | 75.6425s | 0.99075953 |
| 2 | BAAB | 76.3390s | 75.6370s | 0.99080419 |
| 3 | ABBA | 76.3940s | 75.6820s | 0.99067987 |
| 4 | BAAB | 76.3225s | 75.6485s | 0.99116903 |
| 5 | ABBA | 76.3330s | 75.6490s | 0.99103923 |
| 6 | BAAB | 76.3580s | 75.6795s | 0.99111426 |
| 7 | ABBA | 76.2835s | 75.6860s | 0.99216732 |
| 8 | BAAB | 76.3300s | 75.6190s | 0.99068491 |

The geometric mean paired ratio was `0.99105219`, a `0.8948%` speedup. Its
95% confidence interval was `[0.99064263, 0.99146191]`, and the prespecified
one-sided paired t-test on block log ratios returned `p=1.3781645e-10`.
Baseline time was 76.3385s (SD 0.0364s); candidate time was 75.6554s
(SD 0.0365s).

The 16 Stage 3 candidate losses averaged 3.27778750 with sample standard
deviation 0.00177120. The one-sided one-sample t-test against 3.28 returned
`p=7.9708411e-05`, independently confirming the Stage 2 accuracy result.

Block 8's third arm produced its terminal baseline result at 76.349s/loss
3.2806, then took roughly four minutes to finish Inductor process-pool cleanup.
All eight training ranks had produced their result; seven exited immediately,
while one rank waited for four fork children. `py-spy` showed those children in
Python 3.12 `multiprocessing.Process._after_fork`, invoking
`concurrent.futures.process.weakref_cb` on an inherited shutdown lock. The
processes recovered and torchrun exited zero before the fixed ten-minute
watchdog, so the arm remained valid. The delay occurred after the script's
reported training time and no outcome was replaced or rerun.

During Stage 3, upstream master advanced from the measured base `003ff3e` to
`bc1b58e`. Inspection showed the only new change was a README leaderboard
update; `train_gpt.py` and the measured baseline were unchanged. That update
listed merged PR #317 at 1.266 minutes. PR #317's training change is commit
`0016a3c`, already an ancestor of the measured `003ff3e` baseline.
