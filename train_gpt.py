"""NanoGPT speedrun, track 1: train a GPT-2 small-scale model to 3.28 FineWeb val loss on 8xH100.

Launch: torchrun --standalone --nproc_per_node=8 train_gpt.py, after building the exact-match extension once
(`pip install ./exact_match`, needs a Rust toolchain).

This file is the outline of the whole run. The model, optimizer, data and schedules live in
the `track_1_short/` package; `track_1_short/perf/` holds the kernels and precision tricks that make it
fast, and can be skipped when reading for the algorithm.
"""
import os
import sys

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from track_1_short.run_log import log_environment, read_source, start_run_log

# Read the source ASAP, for the run log.
code = read_source(sys.argv[0])

import copy
import gc
import glob
import time
from concurrent.futures import Future

import torch

torch.empty(
    1, device=f"cuda:{os.environ['LOCAL_RANK']}", requires_grad=True
).backward()  # prevents a bug on some systems
import torch._dynamo as dynamo
import torch.distributed as dist
from torch import nn

# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
from track_1_short.canonical_mask import BackgroundCanonicalMask
from track_1_short.config import (
    LR_COOLDOWN_FRAC,
    MODEL_DIM,
    SPLIT_EMBED_STAGE,
    TRAINING_STAGES,
    WS_POST_YARN_EXT,
    Hyperparameters,
)
from track_1_short.data import HEADER_BYTES, ScheduledBatches, cu_seqlens_rows, distributed_data_generator, read_val_prefix
from track_1_short.distributed import setup_distributed
from track_1_short.doc_cache import NBINS as DOC_BINS, ORDERS as DOC_ORDERS, DocChain
from track_1_short.exact_counts import NBINS, ORDERS, ChainFit, ExactCounts, chain_prob, components, context_counts, gate_L
from track_1_short.model.gpt import ATTN_BANK_ORDER, FP8_EXACT_SCALE_CALLS, GPT
from track_1_short.model.prefix_prediction import build_prefix_table_bucket
from track_1_short.ngram_table import (
    MAX_CYCLE_STEPS,
    NGRAM_DIM,
    NgramTable,
    is_update_step,
    ngram_row_ids,
)
from track_1_short.perf.cuda_graphs.baked_addresses import assert_addresses_unchanged, baked_addresses
from track_1_short.perf.cuda_graphs.capture_plan import plan_warmup, step_graph_key
from track_1_short.perf.cuda_graphs.fp8_refresh_graphs import Fp8RefreshGraphs
from track_1_short.perf.cuda_graphs.optimizer_graphs import AnvilBankGraphs, assert_banks_alias_live_state
from track_1_short.perf.cuda_graphs.step_graphs import StepGraphs
from track_1_short.perf.deferred_gathers import DEFERRED_LABELS, DeferredGathers
from track_1_short.perf.kernels.mlp import prime_stage_cache
from track_1_short.perf.pinned_batches import PinnedBatchStaging
from track_1_short.perf.row_prefetch import PrefetchStaging, RowPrefetch
from track_1_short.perf.value_embed_pull import ValueEmbedPull
from track_1_short.retrieval import LARGE_HOST, RET_CELLS, FitRows, OnlineCache, ValidationCache, mix_q, pin_cpus, pin_threads, rows_part
from track_1_short.sampled_softmax import SampledSoftmax
from track_1_short.schedule import TrainingSchedule
from track_1_short.tail_average import TailAverages
from track_1_short.training import TrainingManager

# One compiled graph per (stage, sampled-softmax P, train/eval) combination: 6 train graphs (stages 0, 1, 3,
# 4 and stage 2 at two P values), the same 6 under no_grad (the step graphs' self-checks) and the eval
# graph of each validation's window configuration (only the final one in the record run).
dynamo.config.recompile_limit = 64
# Step lines in the timed loop (console and log): every Nth step, plus the last two (record #360).
PRINT_EVERY = 25


def train_step(training_manager, step_graphs, row_prefetch, sampled_softmax, batches, deferred_gathers, online_cache,
               step: int, lm_head_f8_col, prefetch_next: bool):
    """One optimizer step on one microbatch: requantize for the previous update, forward/backward, update.

    The forward and backward are CUDA-graph replays of the compiled model's training loss
    (perf/cuda_graphs/step_graphs.py; the warmup's first visit of each configuration captures them).
    With `prefetch_next`, the next step's sampled-softmax candidates are built on the prep thread
    while this step's optimizer runs. The n-gram rows and value_embeds rows of the step's update cycle
    were pulled into the model's cache and replica at the previous table update; at an update step the
    next cycle's pulls run alongside (perf/row_prefetch.py). The optimizer leaves the bank gathers in
    flight; the next step waits them and refreshes the fp8 weight caches (perf/deferred_gathers.py).
    """
    # Peek, not take: the row pulls for this step's cycle read this batch too. Taken at the end.
    batch = batches.peek(step)
    ngram_table = row_prefetch.ngram
    ngram_slots = row_prefetch.forward_slots(step, batch.ngram_ids)
    # This step's sampled-softmax candidate set starts uploading on its copy stream first, so the copy
    # runs under the flush below (no-op on full-softmax steps).
    sampled_softmax.upload(step, batch.targets_cpu)
    # The last moment before the first reader of the banks: wait the previous step's gathers and
    # refresh the fp8 weights.
    deferred_gathers.flush()
    # Before the forward, after the lm_head fp8 refresh just above: the candidates' lm_head rows. None on
    # full-softmax steps.
    sampled_loss = sampled_softmax.gather(step, lm_head_f8_col)
    step_graphs.forward(step, batch, ngram_slots, training_manager.get_forward_args(sampled_loss),
                        online_cache.rows(step, batch.inputs.numel()))
    # Eager, between the two replays: the next cycle's row-id exchange hides under the backward.
    row_prefetch.after_forward(step)
    ngram_grad = step_graphs.backward()
    ngram_table.accumulate_grad(ngram_slots, step_graphs.hold_ngram_grad(
        ngram_grad, pending=len(ngram_table.pending), event_this_step=row_prefetch.is_update(step)))
    sampled_softmax.mark_readers_done()
    row_prefetch.post_prep(step)
    if prefetch_next:
        sampled_softmax.prefetch(step + 1, batches.peek(step + 1).targets_cpu)
    gathers = training_manager.step_optimizers(step, row_prefetch, DEFERRED_LABELS)
    # lm_head is Adam-updated, i.e. only on Adam steps, so its fp8 copies only change then.
    deferred_gathers.hand_over(gathers, refresh_lm=training_manager.is_adam_step(step))
    batches.take(step)


def main():
    args = Hyperparameters()
    # Before the process group, so that every thread inherits this rank's cores (track_1_short/retrieval.py).
    cpus = pin_cpus(int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]))
    env = setup_distributed()
    if args.train_seed is not None:
        torch.manual_seed(args.train_seed)

    # Tokens per rank of the largest training batch, and of a validation batch.
    max_step_tokens = max(s.batch_size for s in TRAINING_STAGES) // env.world_size
    val_tokens_per_rank = args.val_batch_size // env.world_size
    # Pinned slots for every batch's H2D copies, training and validation (perf/pinned_batches.py), sized
    # for the largest batch per rank either reads.
    batch_tokens = [s.batch_size // env.world_size for s in TRAINING_STAGES] + [val_tokens_per_rank]
    batch_staging = PinnedBatchStaging(max(batch_tokens), max(map(cu_seqlens_rows, batch_tokens)), env.device)

    def train_loader(shard_slots=None, want_shard=None):
        return distributed_data_generator(
            args.train_files, TRAINING_STAGES[0].batch_size, TRAINING_STAGES[0].train_max_seq_len, batch_staging,
            shard_slots=shard_slots, want_shard=want_shard,
        )

    def val_loader(first_tokens=None):
        return distributed_data_generator(args.val_files, args.val_batch_size, -1, batch_staging, align_to_bos=False,
                                          first_tokens=first_tokens)

    training_schedule = TrainingSchedule(
        TRAINING_STAGES, args.num_scheduled_iterations, args.num_extension_iterations, device=env.device,
        cooldown_frac=LR_COOLDOWN_FRAC, split_embed_stage=SPLIT_EMBED_STAGE, ws_post_yarn_ext=WS_POST_YARN_EXT,
    )

    print0, flush_log = start_run_log(env.master_process, args.run_id)
    log_environment(print0, code)
    flush_log()

    ########################################
    #          Model and optimizer         #
    ########################################
    model: nn.Module = GPT(
        vocab_size=50257,
        num_layers=11,
        num_heads=MODEL_DIM // 128,
        head_dim=128,
        model_dim=MODEL_DIM,
        max_seq_len=val_tokens_per_rank,
        ngram_dim=NGRAM_DIM,
        world_size=env.world_size,
        device=env.device,
    ).cuda()
    attn_widths = {layer: (model.attn_qk_dim(layer), model.attn_v_dim(layer)) for layer in ATTN_BANK_ORDER}
    print0(
        f"attention heads={model.num_heads} (qk, v) head widths by layer, in bank order={attn_widths} "
        f"seed={'random' if args.train_seed is None else args.train_seed} "
        f"steps={training_schedule.total_steps} stage boundaries={training_schedule.boundaries}"
    )
    model.cast_matrix_weights_bf16()
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)
    dist.broadcast(model.ngram_sign_pool, 0)  # buffer, not in parameters()
    # The hashed n-gram table: 1/world_size of its rows on each rank, with its own sparse Adam. Not a
    # model parameter, so neither the broadcast above nor the state_dict below ever holds it.
    ngram_table = NgramTable(model.ngram_cache, max_step_tokens=max_step_tokens,
                             max_events=training_schedule.total_steps,
                             rank=env.rank, world_size=env.world_size, device=env.device)
    # The row prefetch's pinned upload rings and prep thread, allocated and started off the clock.
    prefetch_staging = PrefetchStaging(max_ngram_rows=2 * MAX_CYCLE_STEPS * max_step_tokens,
                                       max_value_embed_rows=model.value_embeds.shape[0], device=env.device)
    model.quantize_attn_fp8()
    model.quantize_mlp_fp8(refresh_lm=True)
    # The fused MLP kernel picks its pipeline depth by trial launch, which cannot run under compile.
    prime_stage_cache()

    # Early steps train on a sampled softmax (see track_1_short/sampled_softmax.py).
    sampled_softmax = SampledSoftmax(
        training_schedule, model.vocab_size, model_dim=model.lm_head.in_features, max_rows=max_step_tokens,
        rank=env.rank, world_size=env.world_size, device=env.device,
    )
    print0(f"sampled softmax: P by step change {[(s, sampled_softmax.counts[s]) for s in [0, *sampled_softmax.count_change_steps()]]}")

    uncompiled_model = model  # the CUDA-graph owners read and set attributes on the module itself
    model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)
    # The ANVIL bank updates run through their CUDA graphs once captured (eagerly before).
    optimizer_graphs = AnvilBankGraphs(env.device)
    training_manager = TrainingManager(model, training_schedule, bank_update=optimizer_graphs.update)
    # value_embeds' gradient buffer, row-sparse update and replica pulls, on the n-gram table's cycles.
    value_embed_pull = ValueEmbedPull(model.value_embeds, model.vocab_size, training_manager.optimizer,
                                      rank=env.rank, world_size=env.world_size, device=env.device)
    tail_averages = TailAverages(training_manager.optimizer, env.rank, value_embed_updates=is_update_step,
                                 total_steps=training_schedule.total_steps)

    # CUDA graphs (track_1_short/perf/cuda_graphs/): the training step's forward and backward per
    # configuration, the ANVIL bank updates, and the fp8 weight refresh. All are captured in the warmup
    # below and only replayed on the clock.
    step_keys = [step_graph_key(training_schedule, sampled_softmax.counts, s, env.world_size)
                 for s in range(training_schedule.total_steps)]
    step_graphs = StepGraphs(model, training_manager.optimizer, step_keys, ngram_table, value_embed_pull,
                             max_step_tokens, env.device)
    fp8_refresh = Fp8RefreshGraphs(uncompiled_model)
    deferred_gathers = DeferredGathers(value_embed_pull, fp8_refresh)

    # Exact-match retrieval (track_1_short/retrieval.py), allocated and prefaulted before the clock. The online index
    # plans every step's documents and resolves each step's rows on the clock ahead of training; the validation index
    # over all training shards is built on the clock while training runs, and looked up only after training, when the
    # validation tokens are read.
    retrieval_schedule = [(stage.batch_size // env.world_size, stage.train_max_seq_len)
                          for stage, _ in map(training_schedule.lookup, range(training_schedule.total_steps))]
    online_cache = OnlineCache(args.train_files, retrieval_schedule, env.rank, env.world_size, env.device,
                               lookahead=24)  # jobs 16 steps ahead
    # the 6-token index and the 3- and 2-token ones (20 shards) leave out the shards the loader may train on
    exclude = -(-int(1.5 * sum(training_schedule.lookup(s)[0].batch_size + env.world_size
                               for s in range(training_schedule.total_steps))) // 100_000_000)
    val_cache = ValidationCache(args.train_files, args.val_files, val_tokens_per_rank, args.val_tokens,
                                dist.new_group(backend="gloo"), cpus.val_build, cpus.val_lookup, exclude=exclude)
    lower = [ValidationCache(args.train_files, args.val_files, val_tokens_per_rank, args.val_tokens, dist.new_group(backend="gloo"),
                             cpus.val_build, cpus.val_lookup, exclude=exclude, key=o, shards=20) for o in (3, 2)]
    online_cache.full, online_cache.lower = (val_cache, [(2, lower[0]), (3, lower[1])])
    # exact counts at the validation positions and every 4th fit step's (F per rank), and the chain's fit
    fit_first = training_schedule.boundaries[3][0]  # the fit steps: the taper and the extension
    fit_steps, exact_fit = range(fit_first, training_schedule.total_steps, 4), []
    F = sum(training_schedule.lookup(s)[0].batch_size // env.world_size for s in fit_steps)
    V, S, i = args.val_tokens, F // len(fit_steps), torch.arange(args.val_tokens + env.world_size * F, device=env.device)
    q_chunk = torch.where(i < V, i // val_cache.chunk * val_cache.chunk, V + (i - V) // S * S)  # query segments: chunks, fit steps
    exact = ExactCounts(glob.glob(args.train_files), exclude, env.rank, env.world_size, env.device, q_chunk, pinned=LARGE_HOST)
    K, groups = len(ORDERS), len(ORDERS) * NBINS
    # the hint rows, one link after the corpus orders
    K, groups = K + 1, groups + 4 * RET_CELLS
    # the in-document cache of this rank's validation chunks and kept fit positions
    doc_chain = DocChain(len(val_cache.offsets) * val_cache.chunk, val_cache.chunk, F, F // len(fit_steps), env.device)
    K, groups = K + len(DOC_ORDERS), groups + len(DOC_ORDERS) * DOC_BINS
    chain = ChainFit(F, K, groups, env.device, MODEL_DIM, env.world_size, env.rank)
    online_cache.mix = fit_rows = FitRows(fit_steps, max_step_tokens, env.device)
    # the validation prefix's pinned slot (+2: the loader reads one token past the last target) and the all-ranks flag
    val_slot, post_flag = (torch.empty(args.val_tokens + 2, dtype=torch.uint16, pin_memory=True), torch.ones(1, device=env.device))

    ########################################
    #            Warmup kernels            #
    ########################################
    print0("Compiling model, warming up kernels and capturing CUDA graphs (~7 minutes on first execution)", console=True)
    # Warmup the training kernels, then re-initialize the state so we aren't cheating. The n-gram
    # table (16 GB per rank) is not copied: it starts at zero, so ngram_table.reset() restores it.
    initial_state = dict(model=copy.deepcopy(model.state_dict()),
                         optimizer=training_manager.get_state())  # save the initial state

    # Each stage start and each sampled-softmax count change begins a new compiled graph.
    transition_steps = sorted(set(training_manager.get_transition_steps()) | set(sampled_softmax.count_change_steps()))
    # first and last pair of steps in each transition, plus the first Adam steps after the embed split
    # (a new optimizer path for embed)
    split_step = training_schedule.split_step
    compile_steps = sorted({0, 1} | {s + offset for s in transition_steps for offset in [-2, -1, 0, 1] if s + offset >= 2}
                           | {split_step + offset for offset in [1, 2, 3] if split_step + offset < training_schedule.total_steps})
    # Plus the visits the graph captures and their self-checks need (perf/cuda_graphs/capture_plan.py).
    capture_plan = plan_warmup(compile_steps, step_keys, training_manager.is_adam_step, FP8_EXACT_SCALE_CALLS)
    warmup_steps = capture_plan.steps
    assert split_step in warmup_steps, "the embed split must happen during warmup for the steps after it to be warm"
    sampled_softmax.assert_warmup_covers(warmup_steps)
    print0(f"Sampling steps {warmup_steps} for warmup; {len(capture_plan.visits)} step-graph configurations "
           f"(warmup visits {sorted(capture_plan.visits.values())}); optimizer graphs captured at step "
           f"{warmup_steps[capture_plan.optimizer_capture_at]}, fp8 refresh graphs at step "
           f"{warmup_steps[capture_plan.fp8_capture_at]}", console=True)
    warmup_batches = ScheduledBatches(train_loader(), training_schedule, steps=warmup_steps)
    # The warmup runs the same row prefetch over its (non-consecutive) steps. The tables update on
    # every Adam step there, so every warmup cycle fits the cache; the cadence only changes scalars.
    warmup_prefetch = RowPrefetch(ngram_table, value_embed_pull, warmup_batches, warmup_steps,
                                  training_manager.is_adam_step, prefetch_staging)
    for position, step in enumerate(warmup_steps):
        training_manager.advance_schedule(step)
        deferred_gathers.flush()
        # The tail captures, between steps with nothing in flight (see capture_plan.py for the positions).
        if position == capture_plan.optimizer_capture_at:
            optimizer_graphs.capture([training_manager.optimizer.banks[label] for label in training_manager.work_order
                                      if label in training_manager.optimizer.banks])
        if position == capture_plan.fp8_capture_at:
            fp8_refresh.capture()
        # Warmup steps are not consecutive, so no sampled-softmax build is prefetched.
        train_step(training_manager, step_graphs, warmup_prefetch, sampled_softmax, warmup_batches, deferred_gathers,
                   online_cache, step, uncompiled_model.lm_head_f8_col, prefetch_next=False)
    # and the counts' all-reduce, which NCCL sets up on its first call
    dist.all_reduce(exact.out)  # the counts are still all zeros
    print0("Resetting Model", console=True)
    deferred_gathers.flush_final()  # the reset below requantizes from the restored weights
    warmup_prefetch.close()
    # From here a missing or unchecked graph is fatal: the timed run never captures and never falls back
    # to eager (except the fp8 refresh's exact-scale bootstrap, which the graphs never cover).
    step_graphs.seal()
    optimizer_graphs.seal()
    fp8_refresh.seal()
    torch.cuda.synchronize()
    # Every address a graph baked, before the reset: the reset must restore state in place.
    addresses = baked_addresses(uncompiled_model, training_manager.optimizer, step_graphs)
    model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    training_manager.reset(initial_state["optimizer"])
    ngram_table.reset()  # the shard, its Adam state, the live cycle and any unapplied gradient
    value_embed_pull.reset()  # the live cycle and any unapplied gradient; the replica was restored above
    sampled_softmax.reset()
    del warmup_batches, warmup_prefetch, initial_state
    # The FP8 scales are not in the state_dict: restart them from the restored weights.
    model.rearm_fp8_bootstrap()
    model.quantize_attn_fp8()
    model.quantize_mlp_fp8(refresh_lm=True)
    model.train()
    # Replay safety: nothing a graph baked moved, and the optimizer banks still alias the live state.
    assert_addresses_unchanged(addresses, baked_addresses(uncompiled_model, training_manager.optimizer, step_graphs))
    assert_banks_alias_live_state(training_manager.optimizer)
    del addresses
    print0(f"CUDA graphs sealed: {2 * len(step_graphs.captured)} step, {len(optimizer_graphs.graphs)} optimizer, "
           f"{len(fp8_refresh.captured)} fp8 refresh; addresses unchanged across the reset", console=True)

    ########################################
    #        Training and validation       #
    ########################################
    gc.collect()  # frees the warmup loader's shards first, so the slots below reuse their cached pinned blocks
    # Pinned shard slots for the timed loader, allocated before the clock: a fresh 256 MB cudaHostAlloc on the clock
    # holds the driver lock ~144 ms and stalls the main thread's H2Ds. Three, round-robin: the current shard, the next
    # one loading, and the retired one, a whole shard past its last reader (a fetched batch's docs, the BOS scan).
    shard_tokens = max((os.path.getsize(f) - HEADER_BYTES) // 2 for f in glob.glob(args.train_files))
    shard_slots = [torch.empty(shard_tokens, dtype=torch.uint16, pin_memory=True) for _ in range(3)]
    # a shard is preloaded only if the loader's plan reads it (until planned: always)
    want_shard = lambda j: (online_cache.index.planned_shards() or j + 1) > j
    batches = ScheduledBatches(train_loader(shard_slots, want_shard), training_schedule, steps=range(training_schedule.total_steps),
                               on_fetch=online_cache.submit)
    row_prefetch = RowPrefetch(ngram_table, value_embed_pull, batches, range(training_schedule.total_steps),
                               is_update_step, prefetch_staging)

    # The canonical mask is only needed by the final validation, so rank 0 builds it in a child
    # process while we train, and model.canon_mask stays all-zero == no masking until then, which
    # is what the intermediate validations run with. Only the buffer is allocated here; the build
    # is started below the clock, so its whole cost -- not just its use -- lands in the timed region.
    canon_mask_builder = BackgroundCanonicalMask(model.vocab_size, owner=env.master_process, print0=print0)

    # No cyclic GC inside the timed loop: a multi-ms generation-2 pause on one rank stalls every rank at
    # the next collective. What survives until now is frozen out of all future scans; the garbage the
    # loop makes is collected at each validation, with the clock stopped (record #360).
    gc.collect()
    gc.freeze()
    gc.disable()

    # From here a window change rebuilds only the rotary rows a training step reads; each validation
    # completes the tables first (model/attention.py Yarn; record #360). Warmup rebuilt them whole.
    uncompiled_model.limit_yarn_rebuild(max_step_tokens)

    training_time_ms = 0
    # start the clock, on every rank at once: ranks finish their pre-clock prefaults at different times
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    online_cache.start()
    build_gate = Future()  # the first build waits for step 25
    prev = build_gate  # the 2-, 3- and 6-token index builds one after another
    for cache in lower[::-1] + [val_cache]:
        cache.start(prev)
        prev = cache.built
    # the corpus share loads once the 6-token index is built (the 3-token one on small hosts)
    (val_cache if LARGE_HOST else lower[0]).built.add_done_callback(lambda f: exact.load_corpus())
    canon_mask_builder.start()
    # Prefix-token table build, inside the timed region. The tokenizer was loaded at import
    # (get_encoding is cached in tiktoken's registry), so this pays only the table construction,
    # split across ranks: each builds one token bucket and the max over ranks is the whole table.
    # In-place copy and reduce keep the buffer's tensor identity, which the compiled graph holds.
    model.prefix_table.copy_(build_prefix_table_bucket(model.vocab_size, bucket=env.rank, num_buckets=env.world_size))
    dist.all_reduce(model.prefix_table, op=dist.ReduceOp.MAX)
    # The candidate build maps prefix targets on the host.
    sampled_softmax.set_prefix_table(model.prefix_table.cpu().numpy())
    # begin training
    for step in range(training_schedule.total_steps + 1):
        last_step = (step == training_schedule.total_steps)
        if step == 1:
            pin_threads(cpus, canon_mask_builder.pid)
        if step == 25:
            build_gate.set_result(None)
        training_manager.advance_schedule(step)
        # --------------- VALIDATION SECTION -----------------
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            # The deferred gathers land first: validation reads the banks and, at the last step, ships
            # into them.
            if last_step:
                # the post-training work reordered: the validation rows' lookups and the prefix read first
                val_cache.lookup(online_cache.lower)
                read_val_prefix(args.val_files, val_slot)
                # the queries (validation stream, all ranks' fit steps) counted first, under the rest
                vt = val_slot[:V + 1].view(torch.int16).to(env.device, non_blocking=True).to(torch.int32) & 0xFFFF
                f_tok, f_tgt, f_p, *f_h = (torch.cat(e) for e in zip(*exact_fit))
                doc_chain.launch(vt, val_cache.offsets, f_tok, f_tgt)  # on its own stream, under the counting
                exact.q_tok[:V], exact.q_tgt[:V] = vt[:V], vt[1:]
                dist.all_gather_into_tensor(exact.q_tok[V:], f_tok)
                dist.all_gather_into_tensor(exact.q_tgt[V:], f_tgt)
                launched = exact.loaded.is_set()  # the host work first while the corpus still loads
                if launched:
                    exact.launch()
                # The final fp8 refresh is skipped, not moved off the clock: its one consumer is a
                # training forward, and the run ends after this validation.
                deferred_gathers.flush_final()
                training_manager.apply_final_ws_ext()
                val_batches = [batch for _, batch in zip(range(args.val_tokens // args.val_batch_size), val_loader(val_slot))]
                if not launched:
                    exact.launch()
                # Both on the clock: the wait in case the build is somehow not done, and the copy
                # and broadcast of the result because they are part of the mask's cost.
                canon_mask_builder.wait()
                with torch.cuda.stream(torch.cuda.Stream(env.device)):
                    canon_mask_builder.collect(model.canon_mask)  # on its own stream, under the counting
            else:
                deferred_gathers.flush()
                # On the clock: only the live cycle's rows of each value_embeds replica are current, and
                # validation reads every token (perf/value_embed_pull.py).
                value_embed_pull.gather_replica()
            uncompiled_model.complete_yarn_tables()
            # On the clock (record #360): the table has no replica, so reading the val batches and pulling
            # their n-gram rows is part of the run's cost. The cache holds one batch, so the untimed loop
            # below lands each batch's rows again (same routes, no new exchange) right before its forward.
            assert args.val_tokens % args.val_batch_size == 0
            val_steps = args.val_tokens // args.val_batch_size
            # Training is over: this rank's validation chunks are being looked up on the CPU (started above),
            # beside the row pull below.
            assert last_step, "the validation index is looked up once, after training"
            val_pulls = ngram_table.eval_pulls([batch.ngram_ids for batch in val_batches])
            for pull in val_pulls:
                ngram_table.land(pull)
            # On the clock: evaluate (and keep) the tail-averaged weights, not the final iterate. The ship also gathers
            # value_embeds whole. It comes after the steps that read the GPU back (the pulls).
            tail_averages.ship()
            # training positions only: the counts summed over the ranks, the chain's weights
            dist.all_reduce(exact.out)
            f_rows = rows_part(fit_rows.cell[:fit_rows.n], fit_rows.q[:fit_rows.n])
            fQ, fG, _ = components(*context_counts(exact.out, V + env.rank * F + torch.arange(F, device=env.device)),
                                   f_rows + doc_chain.part())
            exact_lam = chain.em(f_p, fQ, fG)
            gate = chain.gate(f_h[0], f_p, fQ, fG, exact_lam)  # the hidden-state gate
            exact_val = [context_counts(exact.out, torch.arange(o, o + val_cache.chunk, device=env.device)) for o in val_cache.offsets]
            val_rows = val_cache.rows()
            # every rank enters this only with its own preparation done
            dist.all_reduce(post_flag)
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, (batch, pull, rows) in enumerate(zip(val_batches, val_pulls, val_rows)):
                    ngram_table.land(pull)
                    loss = model(batch.inputs, batch.targets, batch.cum_seqlens, ngram_table.slots(pull, batch.ngram_ids),
                                 training_manager.get_forward_args(), ret=torch.as_tensor(rows[:, :3], device=env.device))
                    # the chain: the hint rows' link (q, the target's share of the row's counts), the corpus and document links
                    r = torch.as_tensor(rows, device=env.device)
                    link = rows_part(r[:, 0].long(), mix_q(r, batch.targets))
                    Q, G, _ = components(*exact_val[i], link + doc_chain.part(i))
                    L = gate_L(uncompiled_model.val_hidden[:len(Q)], G, *gate, groups)
                    loss = -torch.log(chain_prob(torch.exp(-loss.float()), Q, L).clamp_min(1e-30)).float()
                    val_loss += loss.mean()
            val_loss /= val_steps
            del val_batches, val_pulls
            dist.reduce(val_loss, 0, op=dist.ReduceOp.AVG)
            print0(f"step:{step}/{training_schedule.total_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
            # The clock is stopped: flush the log and collect the loop's garbage here.
            flush_log()
            gc.collect()
            model.train()
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            # The checkpoint has no n-gram table: it is sharded across ranks and this saves rank 0 only.
            if env.master_process and args.save_checkpoint:
                log = dict(step=step, code=code, model=model.state_dict(), optimizer=training_manager.get_state())
                os.makedirs(f"logs/{args.run_id}", exist_ok=True)
                torch.save(log, f"logs/{args.run_id}/state_step{step:06d}.pt")
            # the last step only has the validation loop, so break to avoid training
            break

        # --------------- TRAINING SECTION -----------------
        if step in fit_steps:  # the kept fit step's tokens
            exact_fit.append((batches.peek(step).inputs.clone(), batches.peek(step).targets.int()))
        train_step(training_manager, step_graphs, row_prefetch, sampled_softmax, batches, deferred_gathers, online_cache,
                   step, uncompiled_model.lm_head_f8_col, prefetch_next=step + 1 < training_schedule.total_steps)
        if step in fit_steps:  # and its model probabilities
            exact_fit[-1] += (torch.exp(-step_graphs.current.tok_loss.detach().float()),
                              uncompiled_model.fit_hidden[:S].clone())
            fit_rows.collect(step, exact_fit[-1][1])  # its hint rows' cells and target shares
        tail_averages.tick(step)
        if step + 1 < training_schedule.total_steps:  # the loader 16 batches ahead
            batches.peek(min(step + 16, training_schedule.total_steps - 1))

        # logging, thinned to every PRINT_EVERY steps and the last two
        if (step + 1) % PRINT_EVERY == 0 or step + 1 >= training_schedule.total_steps - 1:
            approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
            print0(f"step:{step+1}/{training_schedule.total_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

    gc.enable()
    if args.run_evals:
        model.eval()
        from evals import hellaswag

        def ngram_cache_slots(inputs):
            """hellaswag's get_bigram_hash hook: pulls the tokens' n-gram rows into the model's cache and
            returns their cache slots, which the forward takes as bigram_input_seq. A collective: every
            rank must score the same number of sequences."""
            return ngram_table.load_eval_batch(ngram_row_ids(inputs).to(env.device))

        hellaswag.evaluate(model=model, schedule_cfg=training_manager.get_forward_args(),
                           seq_len=val_tokens_per_rank, get_bigram_hash=ngram_cache_slots, print0=print0)

    print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
           f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
