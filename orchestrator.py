"""
Experiment orchestrator for the VLA policy-server benchmark.

Given one experiment config -- (model, benchmark, num_clients, serving,
control_hz, inference_steps) -- this:

  1. picks a free port and launches one policy_server.py process, sized to stop
     itself after exactly num_clients * inference_steps requests;
  2. waits for the server to finish loading + warming up (by polling the port);
  3. launches num_clients robot_client.py processes concurrently (each its own
     process, distinct --client-id, each writing a per-request CSV);
  4. waits for every client to exit and the server to finalize;
  5. reads back the server + client CSVs and prints a consolidated summary.

All artifacts for a run land in one directory:
  <out-dir>/<run-name>/
    config.json        the exact config used
    server.csv         per-request server-side metrics
    clientNN.csv       per-request client-side (end-to-end) metrics
    logs/*.log         stdout/stderr of every process

This is closed-loop load: each client sends an observation, waits for its chunk,
plays it back at control_hz, then sends the next. The offered concurrency is set
by num_clients; how the server absorbs it is set by --serving (+ --batch-max).

Run:
  python orchestrator.py --model smolvla --benchmark 2_cameras \
      --num-clients 4 --serving batching --batch-max 4 --control-hz 30
"""

import argparse
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

from robot_client import MAX_INFERENCE_STEPS

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["pi0", "smolvla"]
BENCHMARKS = ["2_cameras"]

DEFAULT_COMPILE_MODE = "max-autotune-no-cudagraphs"
# Arms that can run each batch at its own size (--armory-unpadded).
UNPADDABLE = ("armory", "edf", "batching", "prio-streams", "aegis", "aegis-pnm", "armory-pnm")

# The server catches SIGTERM and writes its CSVs from its shutdown path, so
# terminate() is a request to finalize, not the end of the run. Killing it sooner
# discards the whole run's server-side data.
SERVER_FINALIZE_GRACE_S = 60.0


def find_free_port():
  """Ask the OS for an unused localhost port (avoids stale-server collisions)."""
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def wait_for_port(host, port, timeout, server_proc):
  """Block until host:port accepts a TCP connection (the server binds it only
  after model load + warmup), or raise on timeout / early server exit."""
  deadline = time.time() + timeout
  while time.time() < deadline:
    if server_proc.poll() is not None:
      raise RuntimeError(f"server exited early with code {server_proc.returncode}")
    try:
      with socket.create_connection((host, port), timeout=1.0):
        return
    except OSError:
      time.sleep(0.5)
  raise TimeoutError(f"server not ready on {host}:{port} within {timeout}s")


def visible_device():
  """The CUDA device this run uses (CUDA_VISIBLE_DEVICES, first entry)."""
  return (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip()


def wait_for_gpu_free(pids, timeout=90.0):
  """Block until none of our pids still hold memory on this run's GPU.

  The driver frees a CUDA context asynchronously, so a reaped child's allocation
  can still be held when the next run starts loading and OOMs. Best effort."""
  ours = {str(p) for p in pids}
  try:
    uuid = subprocess.run(
      ["nvidia-smi", "-i", visible_device(), "--query-gpu=uuid", "--format=csv,noheader"],
      capture_output=True, text=True, timeout=10).stdout.strip()
  except (OSError, subprocess.SubprocessError):
    return
  start = time.time()
  while True:
    try:
      out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
      return
    held = []
    for line in out.splitlines():
      f = [x.strip() for x in line.split(",")]
      if len(f) == 3 and f[0] in ours and (not uuid or f[1] == uuid):
        held.append(f"pid {f[0]}: {f[2]} MiB")
    waited = time.time() - start
    if not held:
      if waited > 1.0:
        print(f"[orchestrator] GPU drained in {waited:.1f}s")
      return
    if waited >= timeout:
      print(f"[orchestrator] GPU still held after {timeout:g}s by "
            f"{'; '.join(held)}; continuing anyway.")
      return
    time.sleep(1.0)


def physical_cores():
  """[(cpu_id, numa_node)] for one logical CPU per PHYSICAL core we may run on.

  Hyperthread siblings share execution units, so handing two clients a core pair
  is not the isolation we are after. Restricted to sched_getaffinity: lscpu
  reports the whole machine, but taskset rejects CPUs outside a cgroup cpuset."""
  allowed = os.sched_getaffinity(0)
  out = subprocess.run(["lscpu", "-p=CPU,CORE,NODE"],
                       capture_output=True, text=True).stdout
  rows = sorted(tuple(int(x) for x in ln.split(","))
                for ln in out.splitlines() if ln and not ln.startswith("#"))
  seen, phys = set(), []
  for cpu, core, node in rows:
    if cpu in allowed and core not in seen:
      seen.add(core)
      phys.append((cpu, node))
  return phys


def gpu_numa_node():
  """NUMA node of the visible CUDA device, or None. The server should sit on it:
  H2D staging of ~1.5 MB observations crosses the socket interconnect otherwise."""
  try:
    # Ask about the device this run will actually use. This box has two GPUs on
    # different sockets (A2 on node0, V100S on node1), so querying index 0 blindly
    # would pin the server to the wrong socket whenever CUDA_VISIBLE_DEVICES != 0.
    #
    # This is only sound because launch_server forces CUDA_DEVICE_ORDER=PCI_BUS_ID:
    # under CUDA's default FASTEST_FIRST the indices are REORDERED relative to
    # nvidia-smi (here they are exactly reversed), so this lookup would resolve
    # the wrong GPU and pin the server to the wrong socket.
    bus = subprocess.run(["nvidia-smi", "-i", visible_device(),
                          "--query-gpu=pci.bus_id", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=10).stdout.split()[0]
    # nvidia-smi prints 00000000:3B:00.0; sysfs wants 0000:3b:00.0
    bdf = bus.strip().lower()
    bdf = bdf[4:] if bdf.startswith("0000000") else bdf
    node = int(open(f"/sys/bus/pci/devices/{bdf}/numa_node").read().strip())
    return node if node >= 0 else None
  except Exception:
    return None


def plan_cpu_pinning(num_clients, server_cpus=None, client_cpus=None):
  """(server_cpu_list, [per-client cpu]) as taskset strings, or (None, None).

  Clients and the server otherwise share every core: 10-20 client processes each
  generating a 256x256x3 image per request, pickling ~1.5 MB, and running a gRPC
  stack will preempt the server's own unpickling and H2D staging. That contention
  lands in the measured latency AND in the prediction error, because it delays
  when a robot's next request actually reaches the server.

  Default split: the server takes the whole NUMA node its GPU is attached to; the
  clients get one physical core each on the other node."""
  phys = physical_cores()
  if server_cpus and client_cpus:
    given = [c.strip() for c in client_cpus.split(",")]
    return server_cpus, [given[i % len(given)] for i in range(num_clients)]

  gnode = gpu_numa_node()
  nodes = sorted({n for _, n in phys})
  if gnode is None or len(nodes) < 2:
    # Single node (or unknown GPU affinity): reserve a quarter of the cores for
    # the server, floor of 4, and give the rest to clients.
    cpus = [c for c, _ in phys]
    n_srv = min(max(4, len(cpus) // 4), max(1, len(cpus) - 1))  # always leave a client core
    srv, cli = cpus[:n_srv], cpus[n_srv:]
  else:
    srv = [c for c, n in phys if n == gnode]
    cli = [c for c, n in phys if n != gnode]
  if not cli:
    print(f"[orchestrator] WARNING: only {len(phys)} usable core(s); pinning disabled.")
    return None, None
  if num_clients > len(cli):
    print(f"[orchestrator] WARNING: {num_clients} clients for {len(cli)} client "
          f"cores; sharing cores round-robin -- results are NOT contention-free.")
  return ",".join(str(c) for c in srv), [str(cli[i % len(cli)]) for i in range(num_clients)]


# One math thread per process. Without this every client's torch/OpenMP pool
# defaults to the core count (40 here), so a 20-robot fleet asks for ~800 threads
# on 40 cores and the scheduler thrashes -- which shows up as latency the server
# never caused.
_SINGLE_THREAD_ENV = {
  "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}


def launch_server(args, port, expected_requests, run_dir, logs_dir, cpus=None):
  """Start the policy server; it self-terminates after expected_requests."""
  log = open(os.path.join(logs_dir, "server.log"), "w")
  cmd = [
    sys.executable, "-u", "policy_server.py",
    "--model", args.model,
    "--benchmark", args.benchmark,
    "--port", str(port),
    "--serving", args.serving,
    "--batch-max", str(args.batch_max),
    "--batch-timeout-ms", str(args.batch_timeout_ms),
    "--crit-batch-max", str(args.crit_batch_max),
    "--crit-gather-ms", str(args.crit_gather_ms),
    "--crit-stream-priority", str(args.crit_stream_priority),
    "--gate-hold-batches", str(args.gate_hold_batches),
    "--gate-jitter-sigmas", str(args.gate_jitter_sigmas),
    "--expected-requests", str(expected_requests),
    "--run-timeout", str(args.run_timeout),
    "--out-dir", run_dir,
    "--run-name", "server",
    "--compile" if args.compile else "--no-compile",
    "--compile-mode", args.compile_mode,
    "--precision", args.precision,
  ]
  if args.pnm_denoise_ms is not None:
    cmd += ["--pnm-denoise-ms", str(args.pnm_denoise_ms)]
  if args.serving in ("armory", "armory-pnm"):
    cmd += ["--armory-crit-weight", str(args.armory_crit_weight)]
  if args.armory_unpadded and args.serving in UNPADDABLE:
    cmd.append("--armory-unpadded")
  env = dict(os.environ)
  # Make CUDA index GPUs the way nvidia-smi does. The default FASTEST_FIRST puts
  # the V100S at index 0 and the A2 at index 1 on this box -- the reverse of
  # nvidia-smi -- so CUDA_VISIBLE_DEVICES would silently select the other card.
  env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
  if cpus:
    cmd = ["taskset", "-c", cpus] + cmd
    # The server DOES want threads (unpickling, H2D staging, pickling replies),
    # just not more than the cores it owns.
    env["OMP_NUM_THREADS"] = str(len(cpus.split(",")))
  proc = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env)
  return proc, log


def assign_duties(num_clients, critical_fraction, seed):
  """Per-client duty ('critical'/'normal') for the fleet. round(fraction*N) robots
  are critical-duty; which ones is a seeded shuffle so the critical robots are not
  correlated with the Poisson arrival order (the first N ids start earliest)."""
  n_crit = int(round(critical_fraction * num_clients))
  n_crit = max(0, min(num_clients, n_crit))
  duties = ["critical"] * n_crit + ["normal"] * (num_clients - n_crit)
  np.random.default_rng(seed).shuffle(duties)
  return duties


def poisson_start_delays(num_clients, arrival_rate, seed):
  """Per-client delays (s) for the FIRST request, modelling robots coming online
  as a Poisson process: inter-arrival gaps are Exponential(1/arrival_rate). The
  first robot is anchored at t=0 (no wasted wall-clock); each subsequent robot's
  start is offset by an exponential gap, so mean spacing is 1/arrival_rate and
  the full fleet spreads over ~ (num_clients-1)/arrival_rate seconds.

  arrival_rate <= 0 disables staggering (all robots start together)."""
  if arrival_rate <= 0 or num_clients <= 1:
    return [0.0] * num_clients
  rng = np.random.default_rng(seed)
  gaps = rng.exponential(1.0 / arrival_rate, size=num_clients - 1)
  return [0.0] + [float(x) for x in np.cumsum(gaps)]


def group_critical_delays(start_delays, duties, group_size, cycle_s):
  """Critical robots start in groups of up to group_size that share one start, so
  each group lands in one critical batch. Groups are phased cycle_s / n_groups
  apart (cycle_s = one critical request period), each at the first such slot no
  earlier than its first member's own Poisson start. Best-effort starts are kept."""
  crit = sorted((d, c) for c, (d, duty) in enumerate(zip(start_delays, duties))
                if duty == "critical")
  if not crit or group_size < 1:
    return list(start_delays)
  groups = [crit[i:i + group_size] for i in range(0, len(crit), group_size)]
  out = list(start_delays)
  t0, step = groups[0][0][0], cycle_s / len(groups)
  for g, members in enumerate(groups):
    k = max(0, int(np.ceil((members[0][0] - t0 - g * step) / cycle_s - 1e-9)))
    for _, c in members:
      out[c] = t0 + k * cycle_s + g * step
  return out


def crit_steps(args):
  n = args.inference_steps if args.crit_inference_steps is None else args.crit_inference_steps
  return min(n, MAX_INFERENCE_STEPS)


def launch_client(args, client_id, port, run_dir, logs_dir, start_delay, barrier_dir,
                  duty="mixed", cpu=None):
  """Start one robot client process."""
  log = open(os.path.join(logs_dir, f"client{client_id:02d}.log"), "w")
  crit = duty == "critical"
  # Every arm uses the same request point per class.
  horizon = args.crit_exec_horizon if crit else args.exec_horizon
  steps = crit_steps(args) if crit else min(args.inference_steps, MAX_INFERENCE_STEPS)
  cmd = [
    sys.executable, "-u", "robot_client.py",
    "--client-id", str(client_id),
    "--model", args.model,
    "--benchmark", args.benchmark,
    "--server-address", f"127.0.0.1:{port}",
    "--control-hz", str(args.crit_control_hz if crit else args.control_hz),
    "--exec-horizon", str(horizon),
    "--chunk-len", str(args.chunk_len),
    "--h-max", str(args.crit_h_max if crit else args.h_max),
    "--inference-steps", str(steps),
    "--client-device", args.client_device,
    "--duty", duty,
    "--csv",
    "--out-dir", run_dir,
    "--run-name", f"client{client_id:02d}",
    "--start-delay", str(start_delay),
    "--barrier-dir", barrier_dir,
  ]
  env = dict(os.environ)
  env.update(_SINGLE_THREAD_ENV)
  if cpu is not None:
    cmd = ["taskset", "-c", str(cpu)] + cmd
  proc = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env)
  return proc, log


def stop_server(proc, grace=SERVER_FINALIZE_GRACE_S):
  """Ask the server to finalize and wait for it to actually do so.

  The wait is what preserves server.csv; it also guarantees the ~29 GB CUDA
  allocation is back before the next run of a sweep starts loading. Escalating to
  kill() after the grace bounds a wedged server, where no handler can run at all."""
  if proc.poll() is not None:
    return
  proc.terminate()
  try:
    proc.wait(timeout=grace)
    print(f"[orchestrator] server finalized (rc={proc.returncode})")
  except subprocess.TimeoutExpired:
    print(f"[orchestrator] server did not finalize within {grace:g}s; killing "
          f"(server.csv may be missing).")
    proc.kill()
    proc.wait()


def release_barrier(barrier_dir, num_clients, client_procs, timeout=300.0):
  """Wait until every client has signalled ready, then write the shared 'go'
  instant (one absolute epoch all clients wake from). Proceeds anyway on timeout
  or if a client has already died, so a stuck client can't hang the run."""
  deadline = time.time() + timeout
  while time.time() < deadline:
    ready = sum(os.path.exists(os.path.join(barrier_dir, f"ready_{c}"))
                for c in range(num_clients))
    alive = sum(p.poll() is None for p in client_procs)
    # Release once every still-alive client is ready (a client that died before
    # signalling is counted as done so it can't stall the barrier forever).
    if ready >= max(alive, 1) or ready >= num_clients:
      break
    time.sleep(0.1)
  n_ready = sum(os.path.exists(os.path.join(barrier_dir, f"ready_{c}"))
                for c in range(num_clients))
  print(f"[orchestrator] barrier: {n_ready}/{num_clients} clients ready; releasing.")
  with open(os.path.join(barrier_dir, "go"), "w") as f:
    f.write(str(time.time()))


def read_csv_floats(path, columns):
  """Load selected numeric columns from a CSV into a dict of numpy arrays."""
  data = {c: [] for c in columns}
  with open(path, newline="") as f:
    for row in csv.DictReader(f):
      for c in columns:
        data[c].append(float(row[c]))
  return {c: np.array(v) for c, v in data.items()}


def summarize(run_dir, num_clients, wall_s):
  """Read the server + client CSVs and print a consolidated experiment summary."""
  server_csv = os.path.join(run_dir, "server.csv")
  print("\n" + "=" * 74)
  print("  EXPERIMENT SUMMARY")
  print("=" * 74)
  print(f"  wall time       : {wall_s:.1f} s")

  if os.path.exists(server_csv):
    s = read_csv_floats(server_csv, [
      "arrival_time_s", "completion_time_s", "queue_wait_ms", "gpu_time_ms",
      "server_total_ms",
    ])
    n = len(s["server_total_ms"])
    span = s["completion_time_s"].max() - s["arrival_time_s"].min()
    throughput = n / span if span > 0 else float("nan")
    tot = s["server_total_ms"]
    print(f"  server requests : {n}")
    print(f"  throughput      : {throughput:.2f} req/s")
    print(f"  avg queue wait  : {s['queue_wait_ms'].mean():.2f} ms")
    print(f"  avg gpu time    : {s['gpu_time_ms'].mean():.2f} ms")
    print(f"  server total    : p50 {np.percentile(tot,50):.1f} | "
          f"p90 {np.percentile(tot,90):.1f} | p99 {np.percentile(tot,99):.1f} ms")
  else:
    print("  server.csv      : MISSING (server may have failed)")

  # End-to-end round-trip latency across all clients.
  rtts = []
  waits = []
  found = 0
  for cid in range(num_clients):
    cpath = os.path.join(run_dir, f"client{cid:02d}.csv")
    if os.path.exists(cpath):
      found += 1
      c = read_csv_floats(cpath, ["infer_rtt_ms", "robot_wait_ms"])
      rtts.extend(c["infer_rtt_ms"].tolist())
      waits.extend(c["robot_wait_ms"].tolist())
  if rtts:
    rtts = np.array(rtts)
    waits = np.array(waits)
    print(f"  clients found   : {found}/{num_clients}")
    print(f"  end-to-end RTT  : p50 {np.percentile(rtts,50):.1f} | "
          f"p90 {np.percentile(rtts,90):.1f} | p99 {np.percentile(rtts,99):.1f} ms")
    # Idle time between finishing a chunk and the next one landing.
    print(f"  robot wait      : mean {waits.mean():.1f} | "
          f"p50 {np.percentile(waits,50):.1f} | p90 {np.percentile(waits,90):.1f} ms")
  else:
    print(f"  clients found   : {found}/{num_clients} (no client CSVs)")
  print(f"  artifacts       : {run_dir}")
  print("=" * 74)


def parse_args():
  parser = argparse.ArgumentParser(description="VLA policy-server experiment orchestrator")
  parser.add_argument("--model", type=str, choices=MODELS, default="smolvla")
  parser.add_argument("--benchmark", type=str, choices=BENCHMARKS, default="2_cameras")
  parser.add_argument("--num-clients", type=int, default=4, help="Concurrent robot client processes")
  parser.add_argument("--serving", type=str,
                      choices=["batching", "prio-streams", "aegis", "aegis-pnm",
                               "armory", "armory-pnm", "edf"],
                      default="batching",
                      help="Server policy. Baselines: 'batching' (dynamic batcher) and "
                           "'prio-streams' (priority routing only -- no prediction, gate, "
                           "GPU exclusion or work stealing). Ours: 'aegis', and 'aegis-pnm' "
                           "(that scheduler with the action expert offloaded to the "
                           "processing-near-memory units). Ported from Armory: 'armory' (Lookahead), "
                           "'armory-pnm' (Lookahead with the offload) and 'edf'")
  parser.add_argument("--armory-crit-weight", type=float, default=5.0,
                      help="armory/armory-pnm: Lookahead weight of critical robots (5 = LA@5)")
  parser.add_argument("--armory-unpadded", action="store_true",
                      help="armory/edf/batching/prio-streams/aegis and both PNM arms: run each batch at its own "
                           "size instead of padding")
  parser.add_argument("--pnm-denoise-ms", type=float, default=None,
                      help="aegis-pnm/armory-pnm: simulated PNM latency of the action expert "
                           "(default: per-model PNM_DENOISE_MS)")
  parser.add_argument("--batch-max", type=int, default=4,
                      help="max requests per normal batch (every pass is padded to it)")
  parser.add_argument("--batch-timeout-ms", type=float, default=400.0,
                      help="deadline (ms) to fire a partial normal batch")
  parser.add_argument("--crit-batch-max", type=int, default=2,
                      help="critical queue: max critical requests per pass")
  parser.add_argument("--crit-gather-ms", type=float, default=10.0,
                      help="critical queue: window to wait for co-arriving criticals")
  parser.add_argument("--gate-hold-batches", type=float, default=2.0,
                      help="aegis arms: gate hold cap in units of one normal batch")
  parser.add_argument("--gate-jitter-sigmas", type=float, default=2.0,
                      help="aegis arms: guard-band width in standard deviations of the "
                           "prediction error (1.0 leaves ~16%% of the early tail exposed, "
                           "2.0 ~1.5%%)")
  parser.add_argument("--crit-stream-priority", type=int, default=0,
                      help="critical queue: CUDA stream priority for the critical worker (-1 = high)")
  parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                      help="torch.compile the server inference path (--no-compile for the naive baseline)")
  parser.add_argument("--compile-mode", type=str, default=DEFAULT_COMPILE_MODE,
                      help="torch.compile mode passed to the server. Default "
                           f"'{DEFAULT_COMPILE_MODE}'; CUDA graphs were measured slower on "
                           "smolvla batching. Pass a value to override.")
  parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "fp32"], default="fp16",
                      help="fp16 = realistic tensor-core path (matches smolvla_base); fp32 = slow reference")
  parser.add_argument("--arrival-rate", type=float, default=0.0,
                      help="Poisson rate (robots/s) for staggering initial requests; "
                           "0 = all robots start together. Mean gap = 1/rate.")
  parser.add_argument("--seed", type=int, default=0, help="RNG seed for the Poisson start delays (reproducible)")
  parser.add_argument("--critical-fraction", type=float, default=0.2,
                      help="Fraction of the fleet that is critical-duty (hospital / industrial "
                           "plant); the rest are normal-duty household robots. 0.2 = 80/20.")
  parser.add_argument("--control-hz", type=float, default=30.0,
                      help="Control frequency of best-effort robots")
  parser.add_argument("--exec-horizon", type=int, default=50,
                      help="Actions a best-effort robot executes before requesting the next "
                           "chunk (50 = the whole chunk, i.e. no buffer)")
  parser.add_argument("--chunk-len", type=int, default=50,
                      help="Actions every robot plays from each chunk (the model still "
                           "computes its full chunk)")
  parser.add_argument("--h-max", type=int, default=50,
                      help="Most actions a best-effort robot plays from one chunk")
  parser.add_argument("--crit-h-max", type=int, default=50,
                      help="Most actions a critical robot plays from one chunk")
  parser.add_argument("--crit-control-hz", type=float, default=50.0,
                      help="Control frequency of critical robots")
  parser.add_argument("--crit-exec-horizon", type=int, default=30,
                      help="Actions a critical robot executes before requesting the next "
                           "chunk; the rest of the chunk is its buffer")
  parser.add_argument("--inference-steps", type=int, default=10, help="Observations per client (capped at MAX_INFERENCE_STEPS)")
  parser.add_argument("--crit-inference-steps", type=int, default=None,
                      help="Observations per critical client (default: --inference-steps)")
  parser.add_argument("--crit-first-cores", action="store_true",
                      help="Give critical robots the first cores of the client core list "
                           "(put the fastest cores first)")
  parser.add_argument("--group-criticals", action=argparse.BooleanOptionalAction, default=True,
                      help="Start critical robots in groups of --crit-batch-max, phased "
                           "apart within one critical request period")
  parser.add_argument("--client-device", type=str, default="cpu", help="Client-side preprocessing device")
  parser.add_argument("--port", type=int, default=None, help="Server port (default: auto-pick a free one)")
  parser.add_argument("--out-dir", type=str, default="results-exp", help="Root directory for run artifacts")
  parser.add_argument("--pin-cpus", action=argparse.BooleanOptionalAction, default=True,
                      help="Pin the server to its GPU's NUMA node and each client to its "
                           "own physical core, and cap every process to one math thread. "
                           "Without this, clients and server share all cores and the "
                           "contention lands in both latency and prediction error")
  parser.add_argument("--server-cpus", type=str, default=None,
                      help="Override the server's taskset CPU list (e.g. '20-39')")
  parser.add_argument("--client-cpus", type=str, default=None,
                      help="Override the client core pool, comma-separated (e.g. '0,1,2,3'); "
                           "assigned round-robin, one per client")
  parser.add_argument("--run-timeout", type=float, default=900.0,
                      help="Seconds to wait for the fleet to finish once it is running, "
                           "measured from the barrier release. Bounds a wedged run so the "
                           "sweep loses one cell instead of stalling; also forwarded to the "
                           "server, which writes partial results on expiry")
  parser.add_argument("--server-ready-timeout", type=float, default=1200.0,
                      help="Seconds to wait for the server to load + warm up "
                           "(first-time torch.compile max-autotune can take many minutes; "
                           "later runs reuse the Inductor disk cache and are fast)")
  return parser.parse_args()


def main():
  # sweep_experiments.sh wraps this in `timeout`, which sends SIGTERM on its outer
  # deadline. Default-handled, that skips the cleanup below and orphans the whole
  # fleet plus a server holding the GPU, so the NEXT run OOMs. Raise instead.
  interrupted = []

  def raise_on_signal(signum, _frame):
    if interrupted:
      return  # already unwinding; let the cleanup finish
    interrupted.append(signum)
    raise KeyboardInterrupt(signal.Signals(signum).name)

  signal.signal(signal.SIGTERM, raise_on_signal)

  args = parse_args()
  steps = min(args.inference_steps, MAX_INFERENCE_STEPS)
  port = args.port or find_free_port()

  start_delays = poisson_start_delays(args.num_clients, args.arrival_rate, args.seed)
  duties = assign_duties(args.num_clients, args.critical_fraction, args.seed)
  n_crit = duties.count("critical")
  expected_requests = n_crit * crit_steps(args) + (args.num_clients - n_crit) * steps
  if args.group_criticals:
    # Groups spread over one critical request period: its request point, or half a
    # chunk when the request point falls inside the round trip (it then asks
    # again on every reply).
    period_s = max(args.crit_exec_horizon, args.crit_h_max / 2) / args.crit_control_hz
    start_delays = group_critical_delays(start_delays, duties, args.batch_max, period_s)

  # Tag the run by serving policy so runs are distinguishable and never collide.
  serve_tag = {"batching": f"b{args.batch_max}x1",
               "prio-streams": f"priobase{args.batch_max}",
               "aegis": f"aegis{args.batch_max}",
               "aegis-pnm": f"aegispnm{args.batch_max}",
               "armory": f"armory{args.armory_crit_weight:g}_{args.batch_max}",
               "armory-pnm": f"armorypnm{args.armory_crit_weight:g}_{args.batch_max}",
               "edf": f"edf{args.batch_max}"}[args.serving]
  unpadded = args.armory_unpadded and args.serving in UNPADDABLE
  if unpadded:
    serve_tag += "u"
  # Our scheduler's mechanisms are all-or-nothing and tied to the policy itself.
  ours = args.serving in ("aegis", "aegis-pnm")
  # Worker layout the server will actually build (see policy_server.serve).
  scheduled = args.serving in ("prio-streams", "aegis", "aegis-pnm")
  arrival_model = "poisson" if args.arrival_rate > 0 else "simultaneous"
  arr_tag = f"ar{args.arrival_rate:g}" if args.arrival_rate > 0 else "ar0"
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  run_name = (f"{args.model}_{args.benchmark}_{args.num_clients}c_"
              f"cf{args.critical_fraction:g}_{serve_tag}_{arr_tag}_"
              f"{args.control_hz:g}hz_c{args.crit_control_hz:g}hz_s{args.seed}_{timestamp}")
  run_dir = os.path.join(HERE, args.out_dir, run_name)
  logs_dir = os.path.join(run_dir, "logs")
  barrier_dir = os.path.join(run_dir, "barrier")
  os.makedirs(logs_dir, exist_ok=True)
  os.makedirs(barrier_dir, exist_ok=True)

  config = {
    "model": args.model, "benchmark": args.benchmark,
    "num_clients": args.num_clients,
    "control_hz": args.control_hz, "exec_horizon": args.exec_horizon,
    "chunk_len": args.chunk_len,
    "h_max": args.h_max, "crit_h_max": args.crit_h_max,
    "crit_control_hz": args.crit_control_hz, "crit_exec_horizon": args.crit_exec_horizon,
    "inference_steps": steps, "crit_inference_steps": crit_steps(args),
    "group_criticals": args.group_criticals,
    "expected_requests": expected_requests, "port": port,
    "client_device": args.client_device, "timestamp": timestamp,
    "compile": args.compile, "compile_mode": args.compile_mode,
    "precision": args.precision,
    "serving": args.serving, "batch_max": args.batch_max,
    "num_batchers": 1, "critical_workers": 1 if scheduled else 0,
    "pnm_denoise_ms": args.pnm_denoise_ms,
    "crit_batch_max": args.crit_batch_max,
    "crit_gather_ms": args.crit_gather_ms,
    "gate": ours,
    "gate_hold_batches": args.gate_hold_batches,
    "gate_jitter_sigmas": args.gate_jitter_sigmas,
    "crit_stream_priority": args.crit_stream_priority,
    "pin_cpus": args.pin_cpus, "crit_first_cores": args.crit_first_cores,
    "gpu_exclusive": ours,
    "predictor": ours, "work_stealing": ours,
    "armory_crit_weight": (args.armory_crit_weight
                           if args.serving in ("armory", "armory-pnm") else None),
    "armory_unpadded": unpadded,
    "batch_timeout_ms": args.batch_timeout_ms,
    "arrival_model": arrival_model,
    "arrival_rate": args.arrival_rate,
    "seed": args.seed,
    "critical_fraction": args.critical_fraction,
    "duties": duties,
    "start_delays_s": [round(d, 4) for d in start_delays],
  }
  with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

  print(f"[orchestrator] run: {run_name}")
  print(f"[orchestrator] config: {json.dumps(config)}")

  wall_start = time.time()
  server_cpus, client_cpu = (plan_cpu_pinning(args.num_clients, args.server_cpus,
                                              args.client_cpus)
                             if args.pin_cpus else (None, None))
  if client_cpu and args.crit_first_cores:
    # The client core list is ordered fastest first on hybrid CPUs: hand its head to
    # the critical robots so their observations are not built on efficiency cores.
    order = ([c for c in range(args.num_clients) if duties[c] == "critical"]
             + [c for c in range(args.num_clients) if duties[c] != "critical"])
    by_client = [None] * args.num_clients
    for core, c in zip(client_cpu, order):
      by_client[c] = core
    client_cpu = by_client
  if server_cpus:
    print(f"[orchestrator] CPU pinning: server -> {server_cpus} | "
          f"clients -> {','.join(client_cpu)}")
  server_proc, server_log = launch_server(args, port, expected_requests, run_dir,
                                          logs_dir, cpus=server_cpus)
  client_procs, client_logs = [], []
  try:
    print(f"[orchestrator] waiting for server on port {port} (loading model + warmup)...")
    wait_for_port("127.0.0.1", port, args.server_ready_timeout, server_proc)
    print("[orchestrator] server ready; launching clients...")

    if args.arrival_rate > 0:
      print(f"[orchestrator] Poisson arrivals (rate={args.arrival_rate}/s): "
            f"start delays = {[round(d, 3) for d in start_delays]}")
    print(f"[orchestrator] fleet duty: {n_crit} critical / "
          f"{args.num_clients - n_crit} normal (critical_fraction={args.critical_fraction})")
    for cid in range(args.num_clients):
      proc, log = launch_client(args, cid, port, run_dir, logs_dir,
                                start_delays[cid], barrier_dir, duties[cid],
                                cpu=client_cpu[cid] if client_cpu else None)
      client_procs.append(proc)
      client_logs.append(log)

    # Release all clients from one shared instant so the Poisson stagger is
    # measured from a common origin, free of per-client build-time skew.
    release_barrier(barrier_dir, args.num_clients, client_procs)

    # Wait for all clients to finish their inference steps. One ABSOLUTE deadline
    # for the whole fleet, not a per-client timeout: these are waited in sequence,
    # so a per-client value would multiply by num_clients. A client that overruns
    # it is killed so the run can be scored and the sweep can continue.
    deadline = time.time() + args.run_timeout
    for cid, proc in enumerate(client_procs):
      try:
        rc = proc.wait(timeout=max(1.0, deadline - time.time()))
      except subprocess.TimeoutExpired:
        print(f"[orchestrator] client {cid:02d} exceeded the {args.run_timeout:g}s "
              f"run timeout; killing.")
        proc.kill()
        rc = proc.wait()
      status = "ok" if rc == 0 else f"FAILED (rc={rc})"
      print(f"[orchestrator] client {cid:02d} done: {status}")

    # Server self-terminates after expected_requests; give it a moment. Still up
    # means it never reached that count (e.g. a client died), so ask it to
    # finalize and wait for its partial results.
    try:
      server_proc.wait(timeout=30)
      print(f"[orchestrator] server done (rc={server_proc.returncode})")
    except subprocess.TimeoutExpired:
      print(f"[orchestrator] server still running; asking it to finalize "
            f"(up to {SERVER_FINALIZE_GRACE_S:g}s)...")
      stop_server(server_proc)
  finally:
    # Reap EVERY child before returning, and wait for each one to actually be
    # gone. A server still holding its CUDA context still holds its whole
    # allocation -- ~29 GB for pi0 -- so the next run of a sweep starts loading
    # into a card that is not really free and dies with an OOM that reads like
    # its own fault. terminate()/kill() only deliver the signal; the wait() is
    # what guarantees the memory is back.
    stop_server(server_proc)
    for proc in client_procs:
      if proc.poll() is None:
        proc.terminate()
    for proc in client_procs:
      try:
        proc.wait(timeout=10)
      except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    server_log.close()
    for log in client_logs:
      log.close()
    wait_for_gpu_free([server_proc.pid] + [p.pid for p in client_procs])

  summarize(run_dir, args.num_clients, time.time() - wall_start)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt as e:
    print(f"\n[orchestrator] interrupted ({e}); children reaped.")
    sys.exit(130)
