"""
Robot client for the VLA policy-server benchmark.

Each robot client runs as its own process (the orchestrator spawns many to
simulate concurrent robots). One client's control loop, per inference step:

  1. execute the current action chunk one action at a time via select_action()
     at a fixed control frequency (e.g. 30 Hz), simulating the real control rate;
  2. after exec_horizon actions (not the whole chunk), build a preprocessed
     observation batch (batch dim = 1) and send it to the policy server over
     gRPC, tagged with (client_id, step), from a background thread;
  3. keep executing the chunk's remaining, now stale, actions while the server
     works (it may batch this request with other clients' requests);
  4. swap in the fresh chunk on arrival, dropping the stale tail; if the queue
     drains first the robot sits idle -- that idle span is robot_wait;
  5. increment the step and go back to (1).

Asynchronous inference: every robot requests its next chunk at a FIXED action
exec_horizon of the current one (counted from the chunk's first action), so its
next arrival follows deterministically from the reply time -- which is what the
server's predictive gate schedules around.

Note on select_action(): lerobot's Policy.select_action() does two things --
(a) call the model when its action queue is empty, and (b) pop one action per
control step. Here those halves are split across the wire: (a) runs remotely on
the server (predict_action_chunk over a batch), and (b) runs here, popping one
action per 30 Hz tick from the chunk the server returned.

Communication: gRPC (see grpc_api/inference.proto). One unary Infer call == one
inference step. gRPC is already a lerobot dependency and its process-per-client,
blocking round-trip maps directly onto server-side dynamic batching.
"""

import argparse
import csv
import os
import pickle  # nosec: trusted local benchmark traffic only
import time
from collections import deque, namedtuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import grpc
import numpy as np
import torch

from grpc_api import inference_pb2
from grpc_api import inference_pb2_grpc

# gRPC's default 4 MB message cap is too small for multi-camera image tensors;
# lift both directions to 128 MB (benchmark traffic is local, not adversarial).
MAX_MESSAGE_BYTES = 128 * 1024 * 1024
GRPC_CHANNEL_OPTIONS = [
  ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
  ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
]
# Ceiling on one inference call. Without it a wedged server (or GPU) blocks every
# client forever, the orchestrator waits on those clients forever, and the sweep
# stalls until its wall clock -- losing every run that would have followed. Set
# far above any legitimate latency (pi0 at fleet 40 peaks near 2.8 s) so this can
# only ever fire on a real hang, in which case the client dies, the run is marked
# FAILED, and the sweep moves on.
INFER_TIMEOUT_S = 60.0

# Requests each robot issues per run. Nothing in the loop depends on this value --
# observations are generated fresh per step and the task is fixed per client -- so
# it only sets how long a run lasts. Longer runs shrink the fraction spent ramping
# up and draining, and give the per-robot statistics (critical percentiles, scored
# arrival predictions) more samples, which matters most at small fleet sizes where
# only a couple of robots are on critical duty.
MAX_INFERENCE_STEPS = 200

# LeRobot imports
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference


PRIORITY_NORMAL = 0
PRIORITY_CRITICAL = 1

BENCHMARKS = {
  "2_cameras": {
    "cameras": {
      "observation.images.image": (256, 256),
      "observation.images.wrist_image": (256, 256),
    },
    "state_dim": 8,
    "action_dim": 7,
    # One fine-tuned action expert per task: 8 tasks = 8 experts on the server.
    # Fewer tasks than robots, so a batch normally carries repeats -- several
    # robots on the same task share an expert.
    "tasks": [
      # Household / routine.
      ("pick up the black bowl between the plate and the ramekin and place it on the plate", PRIORITY_NORMAL),
      ("open the top drawer and put the bowl inside", PRIORITY_NORMAL),
      ("put the wine bottle on top of the cabinet", PRIORITY_NORMAL),
      ("turn on the stove and put the moka pot on it", PRIORITY_NORMAL),
      ("push the plate to the front of the stove", PRIORITY_NORMAL),
      # Industrial plant / hospital, safety- or time-critical.
      ("shut the leaking valve on the coolant line", PRIORITY_CRITICAL),
      ("pull the emergency stop lever on the conveyor", PRIORITY_CRITICAL),
      ("hand the surgeon the clamp from the sterile tray", PRIORITY_CRITICAL),
    ],
  },
}


def build_features(spec):
  """Turn a benchmark spec into lerobot input/output PolicyFeature dicts."""
  input_features = {
    cam_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w))
    for cam_key, (h, w) in spec["cameras"].items()
  }
  input_features["observation.state"] = PolicyFeature(
    type=FeatureType.STATE, shape=(spec["state_dim"],)
  )
  output_features = {
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(spec["action_dim"],))
  }
  return input_features, output_features


def build_raw_observation(spec):
  """Random raw observation matching the spec."""
  obs = {
    cam_key: np.random.rand(h, w, 3).astype(np.float32)  # HWC float in [0, 1]
    for cam_key, (h, w) in spec["cameras"].items()
  }
  obs["observation.state"] = np.random.rand(spec["state_dim"]).astype(np.float32)
  return obs


def build_preprocessor(model_choice, benchmark, device):
  """Build only the policy's preprocessor so we can turn 
  raw observations into model-ready, preprocessed tensors"""
  spec = BENCHMARKS[benchmark]
  input_features, output_features = build_features(spec)

  # IDENTITY normalization everywhere -> the processor factories need no
  # dataset_stats, matching the run_benchmark_* setup.
  norm = {
    "VISUAL": NormalizationMode.IDENTITY,
    "STATE": NormalizationMode.IDENTITY,
    "ACTION": NormalizationMode.IDENTITY,
  }

  if model_choice == "pi0":
    config = PI0Config(
      input_features=input_features,
      output_features=output_features,
      image_resolution=(224, 224),  # pi0 requires a square resize target
      device=str(device),
      dtype="float32",
      normalization_mapping=norm,
    )
    preprocess, _ = make_pi0_pre_post_processors(config, dataset_stats=None)

  elif model_choice == "smolvla":
    config = SmolVLAConfig(
      input_features=input_features,
      output_features=output_features,
      device=str(device),
      load_vlm_weights=False,
      normalization_mapping=norm,
      pad_language_to="max_length",
    )
    preprocess, _ = make_smolvla_pre_post_processors(config, dataset_stats=None)

  else:
    raise ValueError(f"Unknown model choice: {model_choice}")

  return preprocess, spec


def sample_task(spec, duty="mixed"):
  """Draw one (instruction, priority) pair uniformly from the benchmark's pool.

  duty pins the robot to one class of work: a critical-duty robot (hospital,
  industrial plant) only draws critical tasks, a normal-duty robot only
  household ones. A client draws ONCE at startup and keeps that task for its
  whole run, so priority is a property of the ROBOT -- which is what lets the
  server predict the class of a client's next request from its first one."""
  pool = spec["tasks"]
  if duty == "critical":
    pool = [t for t in pool if t[1] == PRIORITY_CRITICAL]
  elif duty == "normal":
    pool = [t for t in pool if t[1] == PRIORITY_NORMAL]
  elif duty != "mixed":
    raise ValueError(f"Unknown duty: {duty}")
  return pool[np.random.randint(len(pool))]


def build_observation_batch(preprocess, spec, device, task=None, priority=None):
  """Create one preprocessed observation batch (batch dim = 1) ready to send
  to the policy server."""
  if task is None:
    task, sampled_priority = sample_task(spec)
    if priority is None:
      priority = sampled_priority
  elif priority is None:
    priority = PRIORITY_NORMAL
  raw = build_raw_observation(spec)
  obs_frame = prepare_observation_for_inference(raw, device, task)
  obs = preprocess(obs_frame)
  obs["priority"] = torch.tensor([bool(priority)], dtype=torch.bool)
  return obs


def serialize_observation(obs):
  """Pickle a preprocessed observation dict for the wire. Tensors are moved to
  CPU first so the payload is portable and detached from any CUDA context."""
  cpu_obs = {
    key: (value.detach().cpu() if isinstance(value, torch.Tensor) else value)
    for key, value in obs.items()
  }
  return pickle.dumps(cpu_obs, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_action_chunk(payload):
  """Unpickle the server's reply into (chunk of shape (n_action_steps, action_dim),
  obs_epoch), obs_epoch being the wall-clock instant the chunk's observation
  stands for."""
  actions, obs_epoch = pickle.loads(payload)  # nosec: trusted local benchmark traffic only
  # Accept either (n_action_steps, action_dim) or a batch-of-1 (1, n, action_dim).
  if actions.ndim == 3:
    actions = actions.squeeze(0)
  return actions, obs_epoch


# One in-flight inference request's result.
InferResult = namedtuple(
  "InferResult", "chunk round_trip_ms send_time_s recv_time_s priority horizon obs_perf"
)

# Actions executed per inference step, fixed for every robot and every step. At
# 30 Hz this is exactly 1.0 s of motion, so the server's look-ahead
# (reply_time + horizon/control_hz) is deterministic.
DEFAULT_EXEC_HORIZON_ACTIONS = 30
# Action-chunk length (both policies use n_action_steps=50); the horizon is
# clamped to the real length on arrival.
DEFAULT_CHUNK_LEN = 50


class RobotClient:
  """A single simulated robot: builds observations, calls the policy server, and
  plays back each returned action chunk at a fixed control frequency."""

  def __init__(self, client_id, model, benchmark, server_address, control_hz, device,
               exec_horizon_actions=DEFAULT_EXEC_HORIZON_ACTIONS,
               duty="mixed", chunk_len=DEFAULT_CHUNK_LEN, h_max=None):
    self.client_id = client_id
    self.control_hz = control_hz
    self.control_period = 1.0 / control_hz
    self.device = device
    self.step = 0
    self.exec_horizon_actions = exec_horizon_actions
    self.chunk_len = chunk_len
    # Most actions this robot plays from any one chunk before it stops (Armory's
    # H_max); defaults to the whole chunk.
    self.h_max = min(h_max or chunk_len, chunk_len)
    # perf_counter instant of every action played from the current chunk, so the
    # next chunk can skip what was already played since its observation.
    self._played = []
    # Wall-clock receipt of the previous chunk, reported with the next request:
    # Armory's arms measure this robot's reply delay from it, and every arm reads
    # its absence as the robot's first request.
    self._last_recv_epoch = None

    self.model = model
    self.benchmark = benchmark
    self.preprocess, self.spec = build_preprocessor(model, benchmark, device)

    # One robot, one job: drawn once here and reused for every inference step.
    self.duty = duty
    self.task, self.priority = sample_task(self.spec, duty)
    # Index into the FULL pool: that is the server's expert id.
    self.expert_id = self.spec["tasks"].index((self.task, self.priority))

    # Queue of pending actions from the current chunk. This is the local half of
    # lerobot's select_action(): pop one action per control tick.
    self._action_queue = deque()

    # Per-request client-side metrics (end-to-end view), one row per inference step.
    self.records = []
    self._t0 = None

    self.channel = grpc.insecure_channel(server_address, options=GRPC_CHANNEL_OPTIONS)
    self.stub = inference_pb2_grpc.PolicyInferenceStub(self.channel)
    # Observation build + gRPC call run here so the control loop keeps ticking.
    self._sender = ThreadPoolExecutor(max_workers=1)

  def request_action_chunk(self, step):
    """Build one observation and send it to the server, blocking until the
    server returns this client's action chunk. Runs on the sender thread.

    The horizon ships with the observation: the server needs it (with control_hz)
    to know when this robot will come back, which is what its scheduler plans
    around. It is fixed, so that prediction is exact up to transit and jitter."""
    horizon = self.exec_horizon(self.h_max)
    obs = build_observation_batch(
      self.preprocess, self.spec, self.device, task=self.task, priority=self.priority
    )
    obs["exec_horizon"] = horizon
    obs["control_hz"] = self.control_hz
    obs["expert_id"] = self.expert_id
    obs["h_max"] = self.h_max
    obs["last_recv_epoch"] = self._last_recv_epoch
    obs["send_epoch"] = time.time()   # stamped before pickling, as Armory does
    request = inference_pb2.ObservationRequest(
      client_id=self.client_id,
      step=step,
      payload=serialize_observation(obs),
    )
    send = time.perf_counter()
    reply = self.stub.Infer(request, timeout=INFER_TIMEOUT_S)
    recv = time.perf_counter()
    self._last_recv_epoch = time.time()
    chunk, obs_epoch = deserialize_action_chunk(reply.payload)
    obs_perf = obs_epoch - (time.time() - time.perf_counter())
    return InferResult(
      chunk=chunk,
      round_trip_ms=(recv - send) * 1000.0,
      send_time_s=send - self._t0,
      recv_time_s=recv - self._t0,
      priority=self.priority,
      horizon=horizon,
      obs_perf=obs_perf,
    )

  def submit_request(self, step):
    """Fire off one inference request without blocking the control loop."""
    return self._sender.submit(self.request_action_chunk, step)

  def exec_horizon(self, chunk_len):
    """Actions to execute from this chunk before requesting the next one.

    FIXED, not drawn, so the server's look-ahead is exact up to transit and
    jitter."""
    return int(np.clip(self.exec_horizon_actions, 1, chunk_len))

  def select_action(self):
    """Pop and return the next action to execute (local half of select_action).
    Assumes the queue is non-empty; the control loop guards that."""
    return self._action_queue.popleft()

  def execute_action_chunk(self, chunk, horizon, next_step=None, already=0):
    """Play back a chunk at the control frequency, calling select_action() each
    tick. The request point counts from the chunk's first action, as Armory's
    H_min/H_max do: `already` actions of it were skipped (played from the old
    chunk while this one was computed), so the next request goes out after
    horizon - already more (at once if that is <= 0, and no later than the end of
    the chunk). Playback continues on the chunk's tail until that request lands.
    Returns (pending, n_actions, tail_actions)."""
    self._action_queue.clear()
    self._action_queue.extend(chunk)
    self._played = []
    horizon = min(horizon - already, len(chunk))
    pending = None
    if horizon <= 0:
      if next_step is not None:
        pending = self.submit_request(next_step)
      horizon = 0
    n = 0
    last = False
    while self._action_queue:
      tick_start = time.perf_counter()
      _action = self.select_action()  # in a real robot this drives the motors
      self._played.append(tick_start)
      n += 1
      if n == horizon:
        if next_step is None:
          last = True                      # last step: nothing left to request
        else:
          pending = self.submit_request(next_step)
      # Sleep out the remainder of this control period to hold the target Hz.
      # Every action gets its full tick, so a new chunk starts on the next one.
      remaining = self.control_period - (time.perf_counter() - tick_start)
      if remaining > 0:
        time.sleep(remaining)
      if last or (pending is not None and pending.done()):
        break                              # fresh chunk in hand; drop the stale tail
    return pending, n, max(0, n - horizon)

  def await_action_chunk(self, pending):
    """Block on an in-flight request. The block is robot_wait: the robot has run
    out of actions and is idle. Returns (result, robot_wait_ms)."""
    wait_start = time.perf_counter()
    result = pending.result()
    return result, (time.perf_counter() - wait_start) * 1000.0

  def _await_release(self, start_delay, barrier_dir):
    """Coordinate the first-request time across independent client processes.

    Preprocessor build time varies by several seconds between clients, which
    would swamp sub-second Poisson delays if we just slept after building. So
    instead: signal readiness, wait for the orchestrator's shared 'go' instant
    (an absolute wall-clock epoch, identical for all clients), then sleep until
    go + start_delay. This makes the staggering a true Poisson arrival process
    measured from one common origin (and start_delay=0 => a truly simultaneous
    start, free of build-time skew)."""
    ready = os.path.join(barrier_dir, f"ready_{self.client_id}")
    open(ready, "w").close()
    go_file = os.path.join(barrier_dir, "go")
    deadline = time.time() + 300.0
    while not os.path.exists(go_file):
      if time.time() > deadline:
        print(f"[client {self.client_id:02d}] barrier timed out; starting now")
        return
      time.sleep(0.02)
    with open(go_file) as f:
      go_time = float(f.read().strip())
    target = go_time + start_delay
    now = time.time()
    if target > now:
      if start_delay > 0:
        print(f"[client {self.client_id:02d}] Poisson start: +{start_delay:.3f}s after go")
      time.sleep(target - now)
    elif start_delay > 0:
      print(f"[client {self.client_id:02d}] missed +{start_delay:.3f}s slot "
            f"(build overran by {now - target:.2f}s); starting now")
    # Anchor this client's clock to the shared 'go' instant, mapped onto the local
    # perf_counter. Without it every client times from its own first request, so
    # send_time_s/recv_time_s look like a common timeline and are not one.
    self._t0 = time.perf_counter() - (time.time() - go_time)

  def run(self, num_inference_steps, start_delay=0.0, barrier_dir=None):
    """Run the full loop: fetch a chunk, execute its horizon at the control
    frequency while the next request is in flight, then move to the next step.

    start_delay staggers this robot's FIRST request (set by the orchestrator to
    model a Poisson arrival process) so N robots don't all hit the server at
    once. With barrier_dir set, timing is anchored to a shared release instant
    (see _await_release); otherwise start_delay is applied as a plain sleep."""
    if barrier_dir:
      self._await_release(start_delay, barrier_dir)
    elif start_delay > 0:
      print(f"[client {self.client_id:02d}] Poisson start: waiting "
            f"{start_delay:.3f}s before first request")
      time.sleep(start_delay)
    if self._t0 is None:      # no barrier, or it timed out
      self._t0 = time.perf_counter()
    # Step 0 has nothing to execute yet, so its whole round trip is robot_wait.
    pending = self.submit_request(0)
    for i in range(num_inference_steps):
      result, robot_wait_ms = self.await_action_chunk(pending)
      # The chunk starts at the action the robot was on when its observation was
      # taken, so skip what it has played since, then play up to h_max. (The model
      # computes its full chunk; everything past h_max is discarded.)
      skipped = sum(1 for t in self._played if t >= result.obs_perf)
      result = result._replace(chunk=result.chunk[skipped: self.h_max])
      next_step = i + 1 if i + 1 < num_inference_steps else None
      exec_start = time.perf_counter()
      pending, n_actions, tail_actions = self.execute_action_chunk(
        result.chunk, result.horizon, next_step, already=skipped
      )
      exec_ms = (time.perf_counter() - exec_start) * 1000.0
      self.records.append({
        "client_id": self.client_id,
        "step": self.step,
        "send_time_s": result.send_time_s,
        "recv_time_s": result.recv_time_s,
        "infer_rtt_ms": result.round_trip_ms,  # end-to-end: send -> action chunk back
        "robot_wait_ms": robot_wait_ms,   # idle time before this chunk arrived
        "exec_horizon": result.horizon,   # chunk action at which the next request goes out
        "n_actions": n_actions,
        "tail_actions": tail_actions,     # actions run past the horizon while waiting
        "skipped_actions": skipped,       # already played while this chunk was computed
        "control_hz": self.control_hz,
        "exec_ms": exec_ms,               # wall time playing the chunk at control_hz
        "priority": int(result.priority),  # 1 = critical request, 0 = normal
      })
      print(
        f"[client {self.client_id:02d}] step {self.step:03d} | "
        f"{'CRIT' if result.priority else 'norm'} | "
        f"infer RTT {result.round_trip_ms:7.1f} ms | wait {robot_wait_ms:7.1f} ms | "
        f"executed {n_actions:2d}/{len(result.chunk):2d} actions "
        f"(horizon {result.horizon:2d}, {skipped:2d} skipped, {tail_actions:2d} tail) "
        f"@ {self.control_hz:g} Hz "
        f"in {exec_ms:7.1f} ms"
      )
      self.step += 1

  def write_csv(self, out_dir, run_name=None):
    """Write this client's per-request metrics (end-to-end latencies) to CSV.
    Keyed by (client_id, step) so it joins the server-side CSV."""
    if not self.records:
      return None
    fields = ["client_id", "step", "send_time_s", "recv_time_s", "infer_rtt_ms",
              "robot_wait_ms", "exec_horizon", "n_actions", "tail_actions", "skipped_actions",
              "control_hz", "exec_ms", "priority"]
    seconds_cols = {"send_time_s", "recv_time_s"}
    os.makedirs(out_dir, exist_ok=True)
    if run_name is None:
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      run_name = f"vla_client{self.client_id:02d}_{self.model}_{self.benchmark}_{timestamp}"
    out_path = os.path.join(out_dir, run_name + ".csv")

    def fmt(key, val):
      if isinstance(val, float):
        return f"{val:.6f}" if key in seconds_cols else f"{val:.2f}"
      return val

    with open(out_path, "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fields)
      writer.writeheader()
      for row in self.records:
        writer.writerow({k: fmt(k, row[k]) for k in fields})
    return out_path

  def close(self):
    self._sender.shutdown(wait=True)
    self.channel.close()


def parse_args():
  parser = argparse.ArgumentParser(description="VLA robot client")
  parser.add_argument("--model", type=str, choices=["pi0", "smolvla"], default="smolvla")
  parser.add_argument("--benchmark", type=str, choices=list(BENCHMARKS.keys()), default="2_cameras")
  parser.add_argument("--client-id", type=int, default=0, help="This client's id (assigned by the orchestrator)")
  parser.add_argument("--server-address", type=str, default="127.0.0.1:50051", help="host:port of the policy server")
  parser.add_argument("--control-hz", type=float, default=30.0, help="Control frequency for action playback")
  parser.add_argument("--inference-steps", type=int, default=10,
                      help=f"Observations this client sends (up to {MAX_INFERENCE_STEPS})")
  parser.add_argument("--client-device", type=str, default="cpu", help="Device for client-side preprocessing (robots have no GPU; server runs the model)")
  parser.add_argument("--csv", action="store_true", help="Write this client's per-request metrics to CSV")
  parser.add_argument("--out-dir", type=str, default="results-client", help="Directory for the client CSV")
  parser.add_argument("--run-name", type=str, default=None, help="Optional CSV basename override (without extension)")
  parser.add_argument("--start-delay", type=float, default=0.0,
                      help="Seconds to wait before the first request (Poisson stagger; set by orchestrator)")
  parser.add_argument("--barrier-dir", type=str, default=None,
                      help="Directory for the cross-client release barrier (set by orchestrator)")
  parser.add_argument("--exec-horizon", type=int, default=DEFAULT_EXEC_HORIZON_ACTIONS,
                      help="Actions executed per inference step before the next observation "
                           "is sent. Fixed for every robot and every step, so the server's "
                           "arrival look-ahead is deterministic.")
  parser.add_argument("--h-max", type=int, default=None,
                      help="Most actions played from any one chunk before the robot stops "
                           "(Armory's H_max); default: the whole chunk")
  parser.add_argument("--chunk-len", type=int, default=DEFAULT_CHUNK_LEN,
                      help="Actions the robot plays from each returned chunk (the rest is "
                           "discarded), i.e. lerobot's n_action_steps")
  parser.add_argument("--duty", type=str, choices=["mixed", "normal", "critical"], default="mixed",
                      help="Which task pool this robot draws its (single, fixed) task from; "
                           "set by the orchestrator from --critical-fraction")
  return parser.parse_args()


def main():
  args = parse_args()
  device = torch.device(args.client_device)
  # Each client sends up to MAX_INFERENCE_STEPS observations.
  inference_steps = min(args.inference_steps, MAX_INFERENCE_STEPS)

  print(f"[client {args.client_id}] building '{args.model}' preprocessor for '{args.benchmark}' on {device}...")
  client = RobotClient(
    client_id=args.client_id,
    model=args.model,
    benchmark=args.benchmark,
    server_address=args.server_address,
    control_hz=args.control_hz,
    device=device,
    exec_horizon_actions=args.exec_horizon,
    duty=args.duty,
    chunk_len=args.chunk_len,
    h_max=args.h_max,
  )
  print(f"[client {args.client_id}] duty={args.duty} "
        f"priority={'CRITICAL' if client.priority else 'normal'} task={client.task!r}")
  try:
    client.run(inference_steps, start_delay=args.start_delay, barrier_dir=args.barrier_dir)
  finally:
    if args.csv:
      out_path = client.write_csv(args.out_dir, args.run_name)
      if out_path:
        print(f"[client {args.client_id}] saved {out_path}")
    client.close()


if __name__ == "__main__":
  main()
