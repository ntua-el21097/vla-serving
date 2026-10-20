"""
Policy server for the VLA policy-server benchmark.

The server loads one policy model (chosen by --model + --benchmark) once, then
serves many concurrent robot-client processes over gRPC. Each client sends a
preprocessed observation via a unary Infer call; the server runs
predict_action_chunk() and returns that client's action chunk.

Serving policies:
  batching      baseline: ONE dynamic batcher that coalesces queued requests into
                a single padded forward pass.
  prio-streams  baseline: critical requests get their own queue, worker and CUDA
                stream; nothing else.
  aegis         ours: prio-streams plus arrival prediction, a predictive admission
                gate, GPU exclusion and work stealing.
  aegis-pnm     that same scheduler, but the GPU runs only the VLM prefill and the
                action expert is offloaded to processing-near-memory (PNM) units.
  armory, edf   ported from Armory: one worker choosing which waiting requests
                to serve (Lookahead, or earliest deadline first).
  armory-pnm    Lookahead with the same offload as aegis-pnm.

Every policy serves a per-task action expert per request (hetero_expert.py).

Batch size is the unit of GPU efficiency here: a padded batch of B costs barely
more than a batch of 1, so every policy issues one batched pass rather than B
concurrent batch-1 passes on separate streams.

Design:
  - gRPC handler threads (a thread pool) only (de)serialize and park on an event.
  - Worker threads (one batcher, plus a critical worker under prio-streams and our
    arms; one worker under the Armory arms), each with its own CUDA stream, drain
    the wait queues and issue the forward passes.
  - Per-request metrics (queue wait, GPU time, totals, timestamps) are recorded
    and written to a CSV for downstream plots.

Layout: this file is the entry point (CLI -> model load -> warmup -> workers ->
gRPC -> CSV). The pieces live in vla_serving/:
  execution.py   model setup and every GPU forward path (batch-1, padded batch,
                 prefill-only) + the PNM offload model
  scheduling.py  the serving policy: predictor, gate, GPU guard, worker loops
  serving.py     gRPC servicer, argument parsing, metrics CSV

Run:
  python policy_server.py --model smolvla --benchmark 2_cameras --port 50051 \
      --serving batching --batch-max 4 --expected-requests 80
"""

import json
import os
import queue
import signal
import threading
import time
from concurrent import futures
from datetime import datetime

import grpc
import numpy as np
import torch

from grpc_api import inference_pb2_grpc
from robot_client import GRPC_CHANNEL_OPTIONS

from vla_serving.execution import (
  PNM_DENOISE_MS,
  PNM_STACKS,
  PnmOffload,
  build_warmup_obs_cpu,
  force_static_compile,
  force_static_prefill,
  setup_model,
  warmup_shapes,
  warmup_sizes,
)
from vla_serving.hetero_expert import make_hetero
from vla_serving.scheduling import (
  ArmoryMirror,
  ArrivalPredictor,
  BatchLatency,
  GpuGuard,
  GpuTimeModel,
  HeldBatch,
  NormalGate,
  armory_worker,
  batching_worker,
  critical_worker,
)
from vla_serving.serving import (
  PolicyInferenceServicer,
  parse_args,
  write_prediction_errors,
  write_server_csv,
)

# Reproducible random-init weights.
np.random.seed(42)
torch.manual_seed(42)


def serve():
  args = parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model, spec = setup_model(args.model, args.benchmark, device,
                            compile_model=args.compile, compile_mode=args.compile_mode,
                            precision=args.precision)
  # One expert per task. Swapped in before the first compiled call, so every
  # warmup below already compiles the per-task graph.
  router = make_hetero(model, len(spec["tasks"]), max(args.batch_max, args.crit_batch_max))
  # pi0 gets fp16 via autocast (weights stay fp32); smolvla gets fp16 from real
  # fp16 weights, so it needs no autocast.
  autocast_dtype = (torch.float16
                    if args.model == "pi0" and args.precision == "fp16" and device.type == "cuda"
                    else None)
  warmup_obs_cpu = build_warmup_obs_cpu(args.model, args.benchmark)

  # Three policies share the two-queue / two-stream structure (a critical worker
  # and a normal batcher, one CUDA stream each) and differ only in what runs on
  # top of it:
  #   prio-streams  BASELINE: priority routing and nothing else. No predictor, so
  #                 the critical stream just batches on a stricter fixed deadline;
  #                 no gate, no GPU exclusion, no work stealing.
  #   aegis         OURS: + arrival prediction, the admission gate, GPU exclusion
  #                 and work stealing.
  #   aegis-pnm     OURS, with the action expert moved off the GPU. The queueing,
  #                 gate, exclusion and work-steal are shared with 'aegis'
  #                 verbatim -- the arms differ only in which stages the GPU runs.
  scheduled = args.serving in ("prio-streams", "aegis", "aegis-pnm")
  # The gate is not optional in our scheduler: it is the mechanism that reserves
  # the GPU ahead of a predicted critical arrival, which a non-preempting GPU
  # cannot do any other way. The baseline arm is a separate serving policy rather
  # than a flag, so no run can be half-gated.
  ours = args.serving in ("aegis", "aegis-pnm")
  # External baselines ported from Armory: one worker, batches padded to batch_max, none of
  # the mechanisms above.
  armory = args.serving in ("armory", "edf", "armory-pnm")
  # Run at each batch's own size when asked (Armory's own batching).
  unpadded = args.armory_unpadded and args.serving in ("armory", "edf", "batching",
                                                       "prio-streams", "aegis",
                                                       "aegis-pnm", "armory-pnm")

  offload = None
  measured_pass_s = None   # {"critical": s, "normal": s}, filled by warmup_shapes
  measured_sizes_s = None  # {b: s} for the Armory arms, measured at warmup
  if args.serving in ("aegis-pnm", "armory-pnm"):
    biggest = max(args.batch_max, args.crit_batch_max if args.serving == "aegis-pnm" else 0)
    if biggest > PNM_STACKS:
      raise SystemExit(f"[server] {args.serving}: batch limit {biggest} > {PNM_STACKS} PNM stacks")
    denoise_ms = (args.pnm_denoise_ms if args.pnm_denoise_ms is not None
                  else PNM_DENOISE_MS[args.model])
    # The chunk shape comes from the config rather than a probe pass: running one
    # real predict_action_chunk here would pay pi0's full autotune (~9 min) to
    # build a graph this policy never uses, and would show dynamo a second shape.
    offload = PnmOffload(denoise_ms / 1000.0,
                         (model.config.n_action_steps, spec["action_dim"]),
                         torch.float32)
    print(f"[server] PNM offload: GPU runs vlm_prefill only; action expert "
          f"simulated PNM latency {denoise_ms:.3f} ms, {PNM_STACKS} stacks")
  # aegis/prio-streams: force a static graph only when the two shapes actually
  # differ. With crit_batch_max == batch_max there is ONE shape, so dynamo
  # specializes on its own exactly as in batching mode -- and forcing
  # dynamic=False would instead build a different artifact whose cached
  # split-scan kernel faults on reload. ('aegis-pnm' is deliberately excluded:
  # its GPU arm runs vlm_prefill, not the compiled sample_actions.)
  if args.compile and (unpadded or (args.serving in ("aegis", "prio-streams")
                                    and args.crit_batch_max != args.batch_max)):
    if force_static_compile(model, args.compile_mode):
      print(f"  {args.serving} mode: sample_actions recompiled with dynamic=False")
    if offload is not None and force_static_prefill(model, args.compile_mode):
      print(f"  {args.serving} mode: vlm_prefill recompiled with dynamic=False")
  # 'batching' needs no main-thread warmup: its one batcher compiles the static
  # batch_max graph itself, so dynamo only ever sees one shape. A batch-1 warmup
  # here would add a 2nd shape and force the slow dynamic graph.
  if unpadded:
    # Every size up to the largest batch is its own static graph and costs what it
    # measures (Armory's arms use the measurements; the others only need the graphs).
    measured_sizes_s = warmup_sizes(model, warmup_obs_cpu, device, autocast_dtype,
                                    max(args.batch_max,
                                        args.crit_batch_max if scheduled else 0),
                                    offload=offload)
    if ours:
      measured_pass_s = {"critical": measured_sizes_s[args.crit_batch_max],
                         "normal": measured_sizes_s[args.batch_max]}
    elif armory and offload is not None:
      # Armory's latency is dispatch to reply: the prefill plus the PNM latency.
      measured_sizes_s = {b: s + offload.denoise_s for b, s in measured_sizes_s.items()}
  elif scheduled:
    # Compile BOTH shapes here on the main thread. The two workers warm different
    # shapes, so letting them do it would race the compiler. This also MEASURES
    # each shape's pass cost, which is what the gate is sized from below.
    measured_pass_s = warmup_shapes(model, warmup_obs_cpu, device, autocast_dtype,
                                    args.crit_batch_max, args.batch_max, offload=offload)
  elif armory:
    # One padded shape, so every batch takes that pass (plus the simulated PNM
    # latency under armory-pnm).
    pass_s = warmup_shapes(model, warmup_obs_cpu, device, autocast_dtype,
                           args.batch_max, args.batch_max, offload=offload)["normal"]
    extra = offload.denoise_s if offload is not None else 0.0
    measured_sizes_s = {b: pass_s + extra for b in range(1, args.batch_max + 1)}

  request_queue = queue.Queue()
  crit_queue = queue.Queue() if scheduled else None
  results = []
  results_lock = threading.Lock()
  done_event = threading.Event()
  t0_box = [0.0]  # serving-start clock, set after all workers are warm

  workers = []
  gate = None
  guard = None
  gpu_model = None
  predictor = None      # referenced by the servicer and the stats block in any mode
  steal_queue = None    # normal queue the critical worker may pad its batch from
  held = None           # the normal batch being gathered/held, also stealable
  if scheduled:
    # Two workers, two streams: criticals never queue behind a normal batch.
    if ours:
      predictor = ArrivalPredictor(expiry_s=0.05)
      # Seed the cost model with what warmup just measured on THIS model and
      # device, so the first decisions are not made against a guess.
      gpu_model = GpuTimeModel(defaults={**measured_pass_s, **(measured_sizes_s or {})},
                               sized=unpadded)
      guard = GpuGuard()
      gate = NormalGate(predictor, gpu_model, hold_batches=args.gate_hold_batches,
                        jitter_sigmas=args.gate_jitter_sigmas)
      steal_queue = request_queue
      held = HeldBatch()   # the batch being held is stealable too
    warmup_barrier = threading.Barrier(3)  # crit + batcher + main
    workers.append(threading.Thread(
      target=critical_worker,
      args=(crit_queue, model, device, results, results_lock, t0_box,
            args.expected_requests, done_event, warmup_obs_cpu, warmup_barrier,
            autocast_dtype, args.crit_batch_max, args.crit_gather_ms / 1000.0,
            predictor, gpu_model, args.crit_stream_priority, guard, steal_queue,
            offload, router, held, not unpadded),
      daemon=True,
    ))
    workers.append(threading.Thread(
      target=batching_worker,
      args=(request_queue, model, device, results, results_lock, t0_box,
            args.expected_requests, done_event, warmup_obs_cpu, warmup_barrier,
            autocast_dtype, args.batch_max, args.batch_timeout_ms / 1000.0,
            gate, gpu_model, guard, offload, router, held, not unpadded),
      daemon=True,
    ))
  elif armory:
    warmup_barrier = threading.Barrier(2)   # the worker + main
    workers.append(threading.Thread(
      target=armory_worker,
      args=(request_queue, model, device, results, results_lock, t0_box,
            args.expected_requests, done_event, warmup_barrier, autocast_dtype,
            args.batch_max, BatchLatency(measured_sizes_s),
            ArmoryMirror(), args.serving,
            args.armory_crit_weight, router, offload, not unpadded),
      daemon=True,
    ))
  else:
    warmup_barrier = threading.Barrier(2)   # the batcher + main
    workers.append(threading.Thread(
      target=batching_worker,
      args=(request_queue, model, device, results, results_lock, t0_box,
            args.expected_requests, done_event, warmup_obs_cpu, warmup_barrier,
            autocast_dtype, args.batch_max, args.batch_timeout_ms / 1000.0),
      kwargs={"router": router, "pad": not unpadded},
      daemon=True,
    ))
  for w in workers:
    w.start()

  warmup_barrier.wait()  # block until all workers are warm
  t0_box[0] = time.perf_counter()

  server = grpc.server(
    futures.ThreadPoolExecutor(max_workers=args.max_workers),
    # so_reuseport=0: without this, gRPC defaults to SO_REUSEPORT, so a second
    # (e.g. stale) server can silently co-bind the same port and the kernel
    # splits clients across both -- corrupting benchmark results. Fail loudly
    # with "address already in use" instead.
    options=GRPC_CHANNEL_OPTIONS + [("grpc.so_reuseport", 0)],
  )
  inference_pb2_grpc.add_PolicyInferenceServicer_to_server(
    PolicyInferenceServicer(request_queue, crit_queue, guard, predictor), server
  )
  server.add_insecure_port(f"{args.host}:{args.port}")
  if args.host not in ("127.0.0.1", "localhost", "::1"):
    print(f"[server] WARNING: bound to {args.host}, not loopback. Infer unpickles its "
          f"payload, so anyone who can reach this port can run code as you.")
  server.start()
  if scheduled:
    if ours:
      gate_desc = f"gate band={args.gate_jitter_sigmas:g}sigma"
      gate_desc += f", hold<={args.gate_hold_batches:g}xB"
      gate_desc += (f", measured B={1000*measured_pass_s['normal']:.0f}ms"
                    f" C={1000*measured_pass_s['critical']:.0f}ms")
      gate_desc += f", crit_stream_prio={args.crit_stream_priority}"
      gate_desc += ", gpu-exclusive, work-steal"
    else:
      gate_desc = "no predictor, no gate, no exclusion, no work-steal (baseline)"
    gate_desc += ", unpadded" if unpadded else ""
    stages = ("prefill-gpu + expert-pnm" if offload is not None else "full-gpu")
    policy = (f"{args.serving}(crit batch<={args.crit_batch_max} gather={args.crit_gather_ms:g}ms | "
              f"normal max={args.batch_max} timeout={args.batch_timeout_ms:g}ms | {gate_desc} | "
              f"{stages} | workers=1 crit + 1 normal)")
    tag = (f"aegispnm{args.batch_max}" + ("u" if unpadded else "") if offload is not None
           else f"aegis{args.batch_max}" + ("u" if unpadded else "") if ours
           else f"priobase{args.batch_max}" + ("u" if unpadded else ""))
  elif armory:
    rule = (f"lookahead, critical weight {args.armory_crit_weight:g}"
            if args.serving != "edf" else "earliest deadline first")
    shape = (f"prefill {'unpadded batch<=' if unpadded else 'padded to '}{args.batch_max} + expert-pnm"
             if offload is not None
             else f"unpadded batch<={args.batch_max}" if unpadded
             else f"padded to {args.batch_max}")
    policy = f"{args.serving}({rule} | {shape} | 1 worker)"
    tag = {"armory": f"armory{args.armory_crit_weight:g}_{args.batch_max}",
           "armory-pnm": f"armorypnm{args.armory_crit_weight:g}_{args.batch_max}",
           "edf": f"edf{args.batch_max}"}[args.serving]
    if unpadded:
      tag += "u"
  else:
    policy = (f"batching(max={args.batch_max}, timeout={args.batch_timeout_ms:g}ms"
              + (", unpadded)" if unpadded else ")"))
    tag = f"b{args.batch_max}x1" + ("u" if unpadded else "")
  compile_desc = f"compile={args.compile_mode}" if args.compile else "compile=off"
  print(f"[server] {args.model}/{args.benchmark} listening on {args.host}:{args.port} "
        f"({policy}, {args.precision}, {compile_desc})")

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  base = args.run_name or f"vla_server_{args.model}_{args.benchmark}_{tag}_{timestamp}"
  out_path = os.path.join(args.out_dir, base + ".csv")

  # SIGTERM must reach the finalize path below, not CPython's default handler:
  # that one exits WITHOUT unwinding, so the CSVs would never be written. The
  # orchestrator sends it whenever the server has not self-finalized (a dead
  # client means expected_requests is never reached) -- exactly when the partial
  # results are worth most. Idempotent: a second signal only re-sets the event.
  terminated = []

  def finalize_on_signal(signum, _frame):
    if not terminated:
      terminated.append(signum)
      print(f"\n[server] got {signal.Signals(signum).name}; finalizing.")
    done_event.set()

  for sig in (signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, finalize_on_signal)

  try:
    if args.expected_requests:
      # Bounded wait: a request that never completes must not lose the run's
      # data. On timeout we fall through and write whatever was served.
      finished = done_event.wait(timeout=args.run_timeout)
      if finished and not terminated:
        print(f"[server] served {args.expected_requests} requests; finalizing.")
      else:
        with results_lock:
          n = len(results)
        why = "terminated" if terminated else f"TIMED OUT after {args.run_timeout:g}s"
        print(f"[server] {why} with {n}/{args.expected_requests} requests served; "
              f"finalizing partial results.")
    else:
      done_event.wait()  # serve until signalled; same finalize path
  except KeyboardInterrupt:
    print("\n[server] interrupted; finalizing.")
  finally:
    # Wind the workers down BEFORE snapshotting results: writing while in-flight
    # batches are still appending reads a torn list and can drop rows.
    # Leaving daemon workers mid-CUDA while the interpreter tears down the CUDA
    # context triggers a "terminate called" abort (and a nonzero exit code the
    # orchestrator would misread), so we wind them down explicitly.
    server.stop(grace=1.0).wait(timeout=5.0)
    # One sentinel per consumer of each queue.
    request_queue.put(None)
    if crit_queue is not None:
      crit_queue.put(None)
    for w in workers:
      w.join(timeout=5.0)
    with results_lock:
      results = list(results)
    write_server_csv(results, args, out_path)
    if predictor is not None and predictor.errors:
      write_prediction_errors(predictor.errors, t0_box[0],
                              os.path.join(os.path.dirname(out_path) or ".",
                                           "prediction_errors.csv"))
    if gate is not None or guard is not None:
      stats = {}
      if gate is not None:
        stats.update(gate_deferrals=gate.deferrals, gate_cap_releases=gate.cap_releases)
        # The measured pass costs the gate was sized from, so a run's scheduling
        # decisions can be checked against the machine they were made on.
        if measured_pass_s is not None:
          stats.update(measured_normal_ms=round(1000 * measured_pass_s["normal"], 1),
                       measured_critical_ms=round(1000 * measured_pass_s["critical"], 1))
        stats.update(hold_batches=args.gate_hold_batches,
                     jitter_sigmas=args.gate_jitter_sigmas,
                     learned_normal_ms=round(1000 * gpu_model.get("normal", float("nan")), 1),
                     learned_critical_ms=round(1000 * gpu_model.get("critical", float("nan")), 1))
        if predictor is not None:
          stats.update(predictor_jitter_ms=round(1000 * predictor.jitter, 1),
                       predictor_bias_ms=round(1000 * predictor.bias, 1),
                       predictor_scored=predictor.scored)
      if guard is not None:
        stats.update(guard_blocks=guard.normal_blocks,
                     guard_blocked_ms=round(guard.normal_blocked_ms, 1))
      with open(os.path.join(os.path.dirname(out_path) or ".", "gate_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
      print(f"[server] gate/guard stats: {stats}")


if __name__ == "__main__":
  serve()
