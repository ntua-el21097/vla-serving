"""Front end and results: the gRPC servicer, the CLI, and the metrics CSV.

The servicer is deliberately thin -- it (de)serializes, routes by priority and
parks on an event -- so that no scheduling decision is ever made on a gRPC
handler thread.
"""

import argparse
import csv
import os
import pickle  # nosec: trusted local benchmark traffic only
import time

import grpc
import numpy as np

from grpc_api import inference_pb2
from grpc_api import inference_pb2_grpc
from robot_client import BENCHMARKS
from vla_serving.execution import DEFAULT_COMPILE_MODE, PNM_DENOISE_MS
from vla_serving.scheduling import PendingRequest, request_priority

# Ceiling on parking a gRPC handler thread. These threads are NOT daemons, so a
# request no worker ever completes would block interpreter exit forever. Set well
# above the client's own 60 s timeout: the client normally gives up first, and
# this only fires on a wedged worker.
HANDLER_WAIT_TIMEOUT_S = 120.0


class PolicyInferenceServicer(inference_pb2_grpc.PolicyInferenceServicer):
  """gRPC front end: deserialize the observation, enqueue it for scheduling,
  block until a worker finishes, then return the action chunk.

  With crit_queue set (prio-streams and our arms), critical requests are routed
  to their own queue -- served by a dedicated worker on its own stream -- so they
  never queue behind normal work."""

  def __init__(self, request_queue, crit_queue=None, guard=None, predictor=None):
    self.request_queue = request_queue
    self.crit_queue = crit_queue
    self.guard = guard
    self.predictor = predictor

  def Infer(self, request, context):
    recv_time = time.perf_counter()
    recv_epoch = time.time()
    obs = pickle.loads(request.payload)  # nosec: trusted local benchmark traffic
    priority = request_priority(obs)
    pending = PendingRequest(
      client_id=request.client_id,
      step=request.step,
      obs=obs,
      recv_time=recv_time,
      enqueue_time=time.perf_counter(),
      priority=priority,
      recv_epoch=recv_epoch,
    )
    if self.crit_queue is not None and priority:
      # Tell the guard before enqueueing: normals must stop starting new passes
      # from the moment a critical exists, not just once it reaches the GPU. It
      # must also precede on_arrival, which spends the prediction: in between, the
      # gate sees no critical coming and would release a held batch unguarded.
      if self.guard is not None:
        self.guard.critical_arrived()
      # Score the prediction HERE, at arrival. Doing it in critical_worker after
      # the dequeue folds server queue wait and the gather linger into the
      # residual, so the predictor would learn queueing delay, not round trip.
      if self.predictor is not None:
        self.predictor.on_arrival(request.client_id, recv_time, request.step)
      self.crit_queue.put(pending)
    else:
      self.request_queue.put(pending)
    if not pending.event.wait(timeout=HANDLER_WAIT_TIMEOUT_S):
      context.abort(grpc.StatusCode.DEADLINE_EXCEEDED,
                    f"no worker completed client {request.client_id} step {request.step}")
    # Under an offload the worker released the GPU at prefill end and stamped when
    # the action expert will have finished on the fabric. Hold the reply here, in
    # the handler thread, so the wait costs the GPU nothing.
    remaining = pending.ready_at - time.perf_counter()
    if remaining > 0:
      time.sleep(remaining)
    # Refine the prediction critical_worker recorded at the pass end, anchored at
    # the last instant the server controls: after the offload hold and after this
    # handler has been scheduled, so pickling and wakeup delays are inside the
    # model rather than in the jitter. Safe in closed loop: the client cannot send
    # step k+1 before it has this reply, so on_reply(k) precedes on_arrival(k+1).
    if self.predictor is not None and priority:
      self.predictor.on_reply(request.client_id, time.perf_counter(), obs)
    pending.sent_epoch = time.time()
    return inference_pb2.ActionChunk(
      client_id=request.client_id,
      step=request.step,
      payload=pending.result,
    )


SERVER_CSV_FIELDS = [
  "client_id", "step", "worker_id", "arrival_time_s", "queue_wait_ms",
  "gpu_time_ms", "inference_wall_ms", "server_total_ms", "completion_time_s",
  "action_steps", "batch_size", "priority", "queue", "gate_held_ms", "gate_slack_ms",
  # 0 in every GPU-only policy; the offloaded action expert's cost under the PNM arms.
  "offload_ms", "expert_id",
]
_SECONDS_COLS = {"arrival_time_s", "completion_time_s"}


def write_prediction_errors(errors, t0, out_path):
  """One row per scored prediction: when we said the robot would come back, when
  it actually did, and the gap.

  error_ms < 0 means the robot arrived EARLIER than predicted -- the case that
  defeats the gate, because the GPU was still committed to a normal batch that
  was issued believing there was room. The server's running bias/jitter are two
  EWMA endpoints and cannot answer how often that happens or how far; this can."""
  rows = sorted(errors, key=lambda e: e[3])   # by actual arrival
  os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
  with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["client_id", "step", "predicted_s", "actual_s", "error_ms"])
    for client_id, step, pred, actual in rows:
      w.writerow([client_id, "" if step is None else step,
                  f"{pred - t0:.6f}", f"{actual - t0:.6f}",
                  f"{(actual - pred) * 1000.0:.2f}"])

  err = np.array([(a - p) * 1000.0 for _, _, p, a in rows])
  early = err < 0
  print(f"  prediction error: n={len(err)} | mean {err.mean():.1f} | "
        f"sd {err.std():.1f} | p10 {np.percentile(err, 10):.1f} | "
        f"p90 {np.percentile(err, 90):.1f} ms")
  print(f"  arrived early   : {early.sum()}/{len(err)} ({100 * early.mean():.0f}%), "
        f"worst {err.min():.1f} ms")
  print(f"  saved           : {out_path}")


def write_server_csv(results, args, out_path):
  """Write one row per served request and print a short summary. Rows are sorted
  by arrival so timelines/CDFs reconstruct cleanly downstream."""
  if not results:
    print("[server] no requests served; nothing to write.")
    return

  rows = sorted(results, key=lambda r: r["arrival_time_s"])

  def fmt(key, val):
    if isinstance(val, float):
      return f"{val:.6f}" if key in _SECONDS_COLS else f"{val:.2f}"
    return val

  os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
  with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=SERVER_CSV_FIELDS)
    writer.writeheader()
    for row in rows:
      writer.writerow({k: fmt(k, row[k]) for k in SERVER_CSV_FIELDS})

  # Summary: throughput over the served window + latency percentiles.
  span = rows[-1]["completion_time_s"] - rows[0]["arrival_time_s"]
  throughput = len(rows) / span if span > 0 else float("nan")
  totals = np.array([r["server_total_ms"] for r in rows])
  gpu = np.array([r["gpu_time_ms"] for r in rows])
  waits = np.array([r["queue_wait_ms"] for r in rows])
  batch_sizes = np.array([r.get("batch_size", 1) for r in rows])
  serving = args.serving
  mode = {"batching": f"batching(max={args.batch_max})",
          "prio-streams": f"prio-streams(crit<={args.crit_batch_max}, "
                          f"normal<={args.batch_max}, no gate/exclusion/steal)",
          "aegis": f"aegis(crit<={args.crit_batch_max}, normal<={args.batch_max}"
                   + (", unpadded)" if args.armory_unpadded else ")"),
          "aegis-pnm": f"aegis-pnm(crit<={args.crit_batch_max}, normal<={args.batch_max}, "
                       f"expert offloaded)",
          "armory": f"armory(lookahead, crit weight {args.armory_crit_weight:g}, "
                    + (f"unpadded batch<={args.batch_max})" if args.armory_unpadded
                       else f"padded to {args.batch_max})"),
          "armory-pnm": f"armory-pnm(lookahead, crit weight {args.armory_crit_weight:g}, "
                        f"prefill padded to {args.batch_max}, expert offloaded)",
          "edf": (f"edf(unpadded batch<={args.batch_max})" if args.armory_unpadded
                  else f"edf(padded to {args.batch_max})"),
          }[serving]
  print("\n" + "=" * 72)
  print(f"  SERVER SUMMARY  {args.model}/{args.benchmark}  {mode}")
  print("=" * 72)
  print(f"  requests served : {len(rows)}")
  print(f"  throughput      : {throughput:.2f} req/s")
  print(f"  avg batch size  : {batch_sizes.mean():.2f}")
  print(f"  avg queue wait  : {waits.mean():.2f} ms")
  print(f"  avg gpu time    : {gpu.mean():.2f} ms")
  print(f"  server total    : p50 {np.percentile(totals,50):.2f} | "
        f"p90 {np.percentile(totals,90):.2f} | p99 {np.percentile(totals,99):.2f} ms")
  print(f"  saved           : {out_path}")
  print("=" * 72)


def parse_args():
  parser = argparse.ArgumentParser(description="VLA policy server")
  parser.add_argument("--model", type=str, choices=["pi0", "smolvla"], default="smolvla")
  parser.add_argument("--benchmark", type=str, choices=list(BENCHMARKS.keys()), default="2_cameras")
  parser.add_argument("--host", type=str, default="127.0.0.1",
                      help="Bind address. Loopback by default: Infer unpickles its payload, "
                           "so a routable bind is remote code execution for anyone who can "
                           "reach the port")
  parser.add_argument("--port", type=int, default=50051, help="Bind port")
  parser.add_argument("--serving", type=str,
                      choices=["batching", "prio-streams", "aegis", "aegis-pnm",
                               "armory", "armory-pnm", "edf"],
                      default="batching",
                      help="Serving policy. Baselines: 'batching' (one dynamic batcher) and "
                           "'prio-streams' (priority routing only: critical queue on its own "
                           "stream batching to a stricter deadline, with no prediction, gate, "
                           "GPU exclusion or work stealing). Ours: 'aegis' (adds all four) and "
                           "'aegis-pnm' (that scheduler with the action expert offloaded to the "
                           "near-memory fabric). Ported from Armory: 'armory' (Lookahead), "
                           "'armory-pnm' (Lookahead with the offload) and 'edf' (earliest "
                           "deadline first), all padded to --batch-max")
  parser.add_argument("--armory-crit-weight", type=float, default=5.0,
                      help="armory/armory-pnm: Lookahead weight of critical robots (best-effort = 1); "
                           "5 = LA@5")
  parser.add_argument("--armory-unpadded", action="store_true",
                      help="armory/edf/batching/prio-streams/aegis: run each batch at its own size, "
                           "every size compiled and timed at warmup, instead of padding")
  parser.add_argument("--pnm-denoise-ms", type=float, default=None,
                      help="aegis-pnm/armory-pnm: action-expert latency on the PNM fabric. "
                           f"Defaults per model: {PNM_DENOISE_MS}")
  parser.add_argument("--batch-max", type=int, default=4,
                      help="max requests per normal batch (every pass is padded to it)")
  parser.add_argument("--batch-timeout-ms", type=float, default=400.0,
                      help="max wait (ms) to fill a normal batch before firing a partial one")
  parser.add_argument("--crit-batch-max", type=int, default=2,
                      help="critical queue: max critical requests per pass "
                           "(1 = always batch-1)")
  parser.add_argument("--crit-gather-ms", type=float, default=10.0,
                      help="critical queue: window to wait for co-arriving criticals so "
                           "they share one forward pass")
  parser.add_argument("--gate-hold-batches", type=float, default=2.0,
                      help="aegis arms: gate hold cap in units of one normal batch "
                           "(1.0 = strict break-even)")
  parser.add_argument("--gate-jitter-sigmas", type=float, default=2.0,
                      help="aegis arms: width of the guard band in standard deviations of the "
                           "prediction error (1.0 leaves ~16%% of the early tail unprotected, "
                           "2.0 ~1.5%%)")
  parser.add_argument("--crit-stream-priority", type=int, default=0,
                      help="critical queue: CUDA stream priority for the critical worker "
                           "(-1 = high band on sm70, 0 = same as the normal batcher)")
  # There is deliberately no --no-gate: the gate is the mechanism that reserves a
  # non-preemptible GPU ahead of a predicted critical arrival, so a half-gated
  # 'aegis' run would not be our scheduler. Priority routing on its own is a
  # policy in its own right -- use --serving prio-streams.
  parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                      help="torch.compile the inference path (--no-compile for the naive baseline)")
  parser.add_argument("--compile-mode", type=str, default=DEFAULT_COMPILE_MODE,
                      help="torch.compile mode (applied to both models). Default "
                           f"'{DEFAULT_COMPILE_MODE}'; CUDA graphs were measured slower on "
                           "smolvla batching (unsafe-op partitioning). Pass a value to override.")
  parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "fp32"], default="fp16",
                      help="fp16 = realistic tensor-core path (matches smolvla_base); fp32 = slow reference")
  parser.add_argument("--max-workers", type=int, default=128,
                      help="gRPC handler threads; must exceed the number of concurrent clients")
  parser.add_argument("--expected-requests", type=int, default=None,
                      help="If set, write the CSV and stop once this many requests are served")
  parser.add_argument("--run-timeout", type=float, default=900.0,
                      help="Give up waiting for --expected-requests after this many "
                           "seconds and write partial results rather than hang forever")
  parser.add_argument("--out-dir", type=str, default="results-server",
                      help="Directory for the per-request CSV")
  parser.add_argument("--run-name", type=str, default=None,
                      help="Optional CSV basename override (without extension)")
  return parser.parse_args()
