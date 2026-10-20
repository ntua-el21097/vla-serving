"""Scheduling policy: who runs on the GPU, and when.

The predictive machinery (ArrivalPredictor, GpuTimeModel), the two mechanisms
that protect a critical request (GpuGuard for exclusion, NormalGate for
admission), and the worker loops that drive them.

Both worker loops call into execution.py for the forward pass itself, so the
serving policy can be reasoned about without reading any CUDA code.
"""

import contextlib
import pickle  # nosec: trusted local benchmark traffic only
import queue
import threading
import time
from dataclasses import dataclass, field

import torch

from vla_serving.execution import (
  cudagraph_record_lock,
  cudagraph_thread_init,
  get_stream_context,
  reply_time,
  run_batch_stage,
  run_stage,
)


class ArrivalPredictor:
  """When will each critical robot come back?

  Inference is near-deterministic and so is the client loop. The chunk we reply
  with starts at the instant of its observation, and the robot asks again once it
  reaches action exec_horizon of it. By the reply it has already played
  min(h_max - exec_horizon, control_hz * (t_reply - t_obs)) of those (what was
  left of its old chunk), but never more than the old chunk still held at the
  observation: a robot that ran out stood still and resumes its full interval.
  Its next arrival is
      t_reply + (exec_horizon - played) / control_hz        (at least t_reply)
  Everything rides in the observation the client just sent us. We track only
  critical clients -- they are the ones the scheduler has to protect."""

  def __init__(self, expiry_s=0.05, bias_alpha=0.3, bias_max_s=0.5, jitter_alpha=0.1):
    self._next = {}                # client_id -> predicted arrival (perf_counter)
    self._chunk = {}               # client_id -> (obs_epoch, run-out) of the chunk just replied
    self._prev_out = {}            # client_id -> run-out of the chunk before it
    self._lock = threading.Lock()
    # How far past its predicted arrival (bias included) a robot may be before the
    # prediction is dropped: a late robot must not keep holding the normal queue.
    self.expiry_s = expiry_s
    # The reply is stamped in the gRPC handler just before it is sent, but the
    # robot's clock only starts once that reply is on the wire and unpickled, and
    # its next request has to travel back -- the raw formula runs early by exactly
    # that round trip, and bias learns it. One bias per robot: a shared one is
    # swung by a whole critical group arriving with the same offset and then
    # applied to the other groups. A new robot starts from the fleet's mean.
    self._bias = {}
    self.bias_alpha = bias_alpha
    self.bias_max_s = bias_max_s
    # Spread of the prediction error -- the gate's safety margin. Tracked as an
    # RMS (EWMA of err^2, reported as its root) on a slower alpha than bias, so
    # it estimates the spread rather than echoing the last few samples.
    self.jitter = 0.0
    self.jitter_alpha = jitter_alpha
    self._jitter_sq = 0.0
    # Robots whose last prediction expired: they were off their timing (a stalled
    # client), and so is the arrival after it. Neither may move bias or jitter, or
    # one stall widens the gate's margin for seconds.
    self._skip = set()
    self.scored = 0
    # Every scored prediction: (client_id, step, predicted, actual), all
    # perf_counter seconds. Kept so accuracy can be reported as a distribution
    # instead of the endpoint of an EWMA -- see write_prediction_errors().
    self.errors = []

  def on_arrival(self, client_id, now=None, step=None):
    """The predicted request landed; score the prediction, then spend it.

    err is the prediction error, actual - predicted: negative means the robot
    showed up EARLIER than we expected, which is the case that defeats the gate.
    pred already contains the current bias, so err is added incrementally
    (averaging the bias toward err would converge to half the true offset)."""
    with self._lock:
      pred = self._next.pop(client_id, None)
      if pred is not None:
        now = now or time.perf_counter()
        err = now - pred
        # Keep the raw error too. Two EWMA endpoints cannot answer "how often is
        # a robot early, and by how much" -- and that question is the whole basis
        # for reserving the GPU ahead of an arrival.
        self.errors.append((client_id, step, pred, now))
        if err > self.expiry_s:            # landed after its prediction would have expired
          self._skip.add(client_id)
          return
        if client_id in self._skip:        # the first arrival after an expired one
          self._skip.discard(client_id)
          return
        b = self._bias.get(client_id, self._mean_bias())
        self._bias[client_id] = min(max(b + self.bias_alpha * err, -self.bias_max_s),
                                    self.bias_max_s)
        # How far off the estimate typically lands, either way. The gate needs
        # this: a robot that arrives EARLIER than predicted is the case that
        # gets blocked, so scheduling against the mean is not safe. Capped at the
        # expiry, the gate's own limit, so one sample cannot hold the margin open.
        e = min(abs(err), self.expiry_s)
        a = self.jitter_alpha
        self._jitter_sq = (1 - a) * self._jitter_sq + a * e * e
        self.jitter = self._jitter_sq ** 0.5
        self.scored += 1

  def on_reply(self, client_id, reply_time, obs):
    horizon = obs.get("exec_horizon")
    hz = obs.get("control_hz")
    if not horizon or not hz:
      return None
    horizon, hz = float(horizon), float(hz)
    h_max = float(obs.get("h_max", horizon))
    key = obs.get("obs_epoch", obs["send_epoch"])
    send = key - (time.time() - time.perf_counter())
    with self._lock:
      cur = self._chunk.get(client_id)
      if cur is None or cur[0] != key:   # first reply for this chunk (the handler refines it)
        self._prev_out[client_id] = cur[1] if cur else None
      prev_out = self._prev_out.get(client_id)
      played = 0.0
      if obs.get("last_recv_epoch") is not None:   # step 0 had no chunk to play from
        played = min(h_max - horizon, hz * max(0.0, reply_time - send))
        if prev_out is not None:
          played = min(played, hz * max(0.0, prev_out - send))
      # The robot resumes this chunk at index `played` on the reply and plays to h_max.
      self._chunk[client_id] = (key, reply_time + (h_max - played) / hz)
      predicted = (reply_time + max(0.0, horizon - played) / hz
                   + self._bias.get(client_id, self._mean_bias()))
      self._next[client_id] = predicted
    return predicted

  def runout(self, client_id):
    """When the robot runs out of the chunk it was last replied (perf_counter), or None."""
    with self._lock:
      cur = self._chunk.get(client_id)
      return cur[1] if cur else None

  def _mean_bias(self):
    return sum(self._bias.values()) / len(self._bias) if self._bias else 0.0

  @property
  def bias(self):
    """Fleet mean of the per-robot biases (reporting only)."""
    with self._lock:
      return self._mean_bias()

  def _prune(self, now):
    """Drop predictions more than expiry_s overdue. Caller must hold the lock."""
    for c in [c for c, t in self._next.items() if t < now - self.expiry_s]:
      self._next.pop(c, None)
      self._skip.add(c)

  def earliest(self, now):
    """Soonest predicted critical arrival, INCLUDING one already past due.

    This is the gate's view. A prediction that has come due but whose robot has
    not landed yet still has to keep the GPU clear -- it may arrive at any
    moment -- so it must not be filtered out here; the expiry is what eventually
    bounds that hold."""
    with self._lock:
      self._prune(now)
      return min(self._next.values()) if self._next else None

  def upcoming(self, now, horizon_s):
    """(client_id, predicted arrival) of every critical due before now + horizon_s,
    including ones just past due that have not landed yet (within the expiry)."""
    with self._lock:
      return [(c, t) for c, t in self._next.items()
              if now - self.expiry_s < t <= now + horizon_s]


class GpuTimeModel:
  """EWMA of measured GPU time per dispatch class, so the gate predicts the cost
  of the batch it is about to issue from what this GPU has actually been doing.

  sized: passes run unpadded, so a pass costs what its own size costs, whichever
  worker runs it (same graph, same size). Entries are then kept per size."""

  def __init__(self, alpha=0.3, defaults=None, sized=False):
    self.alpha = alpha
    self._ms = dict(defaults or {})
    self._lock = threading.Lock()
    self.sized = sized

  def cost(self, key, n, default):
    """Seconds for a pass of n: its size's entry when sized, else the class's."""
    if not self.sized:
      return self.get(key, default)
    with self._lock:
      return self._ms.get(n, self._ms.get(key, default))

  def learn(self, key, n, s):
    self.update(n if self.sized else key, s)

  def update(self, key, ms):
    with self._lock:
      prev = self._ms.get(key)
      self._ms[key] = ms if prev is None else (1 - self.alpha) * prev + self.alpha * ms

  def get(self, key, default):
    with self._lock:
      return self._ms.get(key, default)


class GpuGuard:
  """Keeps a critical request from ever sharing SMs with a normal batch.

  Normals may not START a forward pass while a critical is queued or executing;
  they wait on the condition and go as soon as the critical stream is clear. This
  is NOT the predictive gate: that one idles the GPU in anticipation of an
  arrival, whereas here the GPU is busy with the critical, so nothing is wasted.

  Only the forward pass is guarded. Each worker's CPU-side work -- unpickling
  ~1.5 MB observations, H2D staging, pickling chunks back -- still overlaps, and
  that is ~70% of per-pass wall time on this server, so serializing the GPU
  portion costs far less than it appears.

  starve_cap_s bounds the wait so a client that dies mid-flight cannot wedge the
  normal queue forever."""

  def __init__(self, starve_cap_s=1.0):
    self.starve_cap_s = starve_cap_s
    self._cv = threading.Condition()
    self._crit_pending = 0     # criticals queued but not yet served
    self._crit_running = False
    self.normal_blocked_ms = 0.0
    self.normal_blocks = 0

  def critical_arrived(self):
    with self._cv:
      self._crit_pending += 1

  def critical_served(self, n):
    with self._cv:
      self._crit_pending = max(0, self._crit_pending - n)
      self._cv.notify_all()

  @contextlib.contextmanager
  def critical_pass(self):
    with self._cv:
      self._crit_running = True
    try:
      yield
    finally:
      with self._cv:
        self._crit_running = False
        self._cv.notify_all()

  def wait_clear(self, deadline):
    """Block while any critical is queued or on the GPU, until perf_counter
    deadline at the latest. Returns True if this had to wait."""
    start = time.perf_counter()
    with self._cv:
      while self._crit_pending > 0 or self._crit_running:
        if time.perf_counter() >= deadline:
          break
        self._cv.wait(timeout=0.002)
    blocked = (time.perf_counter() - start) * 1000.0
    if blocked > 0.5:
      self.normal_blocked_ms += blocked
      self.normal_blocks += 1
    return blocked > 0.5


class NormalGate:
  """Admission control for the normal batch.

  The GPU does not preempt: once a normal batch is issued, a critical arriving
  mid-flight waits for it to finish. VLA inference is predictable -- a padded
  batch takes a near-constant B -- and the closed-loop client tells us when each
  critical robot comes back, so we can simply refuse to start work we cannot
  finish in time. With B the normal batch cost and A the soonest predicted
  critical arrival, dispatch iff

      t + B <= A

  i.e. the last safe moment to start is A - B. A critical then never queues
  behind a normal batch, so it costs only its own pass C.

  A is widened by the predictor's measured jitter, because a critical arriving
  EARLIER than predicted is the case that gets blocked.

  Unsafe means hold the batch and re-check; the critical lands, gets served on
  its own stream, and the batch goes out right after.

  The hold cap, hold_batches * B, is an anti-starvation backstop only. The
  intended release is the critical landing, or its prediction expiring
  (expiry_s past due); releasing at the cap means the model or prediction was
  wrong, so it is counted separately."""

  def __init__(self, predictor, gpu_model, poll_s=0.002, hold_batches=2.0, jitter_sigmas=2.0):
    self.predictor = predictor
    self.gpu_model = gpu_model
    self.hold_batches = hold_batches
    self.jitter_sigmas = jitter_sigmas
    self.poll_s = poll_s
    self.deferrals = 0
    self.cap_releases = 0   # holds that ended on the backstop, not on safety

  def wait_until_safe(self, held=None):
    """Block until issuing this batch, or the largest part of it that fits, is
    safe (or the hold cap expires). Returns (held_ms, slack_ms, n), n the requests
    to issue. All times are SECONDS -- the gpu_model is fed seconds too, which is
    what makes the inequality meaningful."""
    start = time.perf_counter()
    deferred = False
    while True:
      ok, slack_s, n = self.check(start, deferred, held)
      if ok:
        held_ms = (time.perf_counter() - start) * 1000.0
        return held_ms, (0.0 if slack_s == float("inf") else slack_s * 1000.0), n
      deferred = True
      time.sleep(self.poll_s)

  def check(self, start, deferred, held=None):
    """One decision of wait_until_safe for a hold that began at start: (ok,
    slack_s, n). Counts the hold once it ends, if it deferred at all."""
    now = time.perf_counter()
    # Re-read every pass: critical_worker updates the model mid-hold, and unpadded
    # it can also steal from the held batch, so price the batch at its size now.
    size = max(1, held.size()) if held is not None else 1
    jitter = self.predictor.jitter
    a = self.predictor.earliest(now)
    if a is not None:
      # Protect the EARLY tail, not the mean: a critical that shows up sooner
      # than predicted is the one that gets blocked. One sigma leaves ~16% of
      # that tail exposed; two takes it to ~1.5%.
      a = max(a - self.jitter_sigmas * jitter, now)
    # Unpadded, a smaller batch is a shorter pass: issue the largest part of the
    # held batch that still ends before the critical lands; the rest stays held.
    for n in range(size, 0, -1):
      slack_s = float("inf") if a is None else a - (now + self.gpu_model.cost("normal", n, 0.15))
      if slack_s >= 0:
        break
    b_s = self.gpu_model.cost("normal", size, 0.15)
    if slack_s < 0:   # nothing fits: on a cap release the whole batch goes
      n, slack_s = size, a - (now + b_s)
    # Anti-starvation cap as a multiple of the normal batch cost. Holding costs
    # normals the hold; it saves the critical at most B, the time it would
    # otherwise have spent queued behind this batch. So B is strict break-even
    # and hold_batches is how much normal delay we spend per unit of critical
    # delay saved. A millisecond constant would mean a different number of
    # batches on each model.
    capped = (now - start) >= self.hold_batches * b_s
    ok = slack_s >= 0 or capped
    if ok and deferred:
      self.deferrals += 1
      if capped and slack_s < 0:
        self.cap_releases += 1
    return ok, slack_s, n


class HeldBatch:
  """The normal batch the batcher is gathering or holding, visible to the
  critical worker. Requests the batcher has already dequeued are no longer in
  the queue, so without this a critical pass could not steal them -- and under a
  gate hold that is exactly when they wait longest. Critical passes take from
  here first, then from the queue; the batcher runs whatever is left."""

  def __init__(self):
    self._lock = threading.Lock()
    self._reqs = []

  def add(self, req):
    with self._lock:
      self._reqs.append(req)

  def size(self):
    with self._lock:
      return len(self._reqs)

  def take(self, n):
    with self._lock:
      out, self._reqs = self._reqs[:n], self._reqs[n:]
      return out

  def close(self, n=None):
    """The batcher is admitted: hand it the oldest n (all if None) and stop sharing
    them; the rest stay held for its next batch."""
    with self._lock:
      n = len(self._reqs) if n is None else n
      out, self._reqs = self._reqs[:n], self._reqs[n:]
      return out


@dataclass
class PendingRequest:
  """A request parked in the wait queue until an inference worker serves it."""
  client_id: int
  step: int
  obs: dict
  recv_time: float       # when the gRPC handler received it (server arrival)
  enqueue_time: float    # when it was pushed onto the wait queue
  event: threading.Event = field(default_factory=threading.Event)
  result: bytes = b""    # pickled action chunk, filled in by the worker
  priority: int = 0      # 1 = critical (from obs["priority"]), 0 = normal
  recv_epoch: float = 0.0  # wall clock at arrival / at reply send: Armory's arms
  sent_epoch: float = 0.0  # measure per-robot network delay from these
  ready_at: float = 0.0  # perf_counter instant the reply may be sent. Equals the
                         # GPU pass end except under 'aegis-pnm' serving, where the
                         # offloaded action expert still has to run.


def request_priority(obs):
  """Read the client's priority flag out of a received observation."""
  v = obs.get("priority")
  if v is None:
    return 0
  return int(bool(v.item() if torch.is_tensor(v) else v))


def dispatch_obs_epoch(req, dispatch_epoch):
  """Every arm's chunk starts from the robot's state at dispatch, as Armory's
  robots stream an observation each tick: dispatch time minus this request's own
  robot-to-server delay. Recorded on the obs for the predictor too."""
  epoch = dispatch_epoch - max(0.0, req.recv_epoch - req.obs["send_epoch"])
  req.obs["obs_epoch"] = epoch
  return epoch


class CriticalPlanner:
  """The critical gather's tie-breaker. With criticals waiting and others predicted
  to land during the next pass, either dispatch now (the late ones take the next
  pass) or wait for some of them and serve them together. Each option is scored
  by the starvation it causes -- reply time past each robot's own run-out time --
  and the least is taken; with none either way, the one that batches the most.
  Padded, work stealing fills every pass to the same shape, so a separate pass
  costs no GPU time of its own and earns nothing here; unpadded, each pass is
  priced at its own size (pass_s(n)). Noise: a late request counts from
  when it is usable (predicted arrival + measured unpickling), deadlines carry a
  margin of 2 x the prediction jitter, and the reply's trip back is service.
  Run-outs come from the predictor, which knows where a robot that stood still
  resumed; observation + h_max is only the fallback before its first reply."""

  def __init__(self, alpha=0.1):
    self.alpha = alpha
    self.chunk_start = {}   # client -> perf time its current chunk's action 0 is for
    self.chunk_len_s = {}   # client -> h_max / control_hz
    self.net_s = 0.0        # robot -> server delay (EWMA), stands in for the way back
    self.unpickle_s = 0.0   # arrival -> queued (EWMA)

  def _ewma(self, old, x):
    return x if old == 0.0 else (1 - self.alpha) * old + self.alpha * x

  def observe(self, req):
    """A critical was dispatched: its next chunk starts at its dispatch stamp."""
    self.net_s = self._ewma(self.net_s, max(0.0, req.recv_epoch - req.obs["send_epoch"]))
    self.unpickle_s = self._ewma(self.unpickle_s, max(0.0, req.enqueue_time - req.recv_time))
    self.chunk_start[req.client_id] = req.obs["obs_epoch"] - (time.time() - time.perf_counter())
    self.chunk_len_s[req.client_id] = float(req.obs["h_max"]) / float(req.obs["control_hz"])

  def _runout(self, cid, predictor):
    out = predictor.runout(cid)
    if out is None and cid in self.chunk_start:
      out = self.chunk_start[cid] + self.chunk_len_s[cid]
    return out

  def _starve(self, cid, t, n, pass_s, pnm_s, margin, predictor):
    """How far cid's reply lands past its run-out if its pass of n starts at t."""
    out = self._runout(cid, predictor)
    if out is None:
      return 0.0
    service = pass_s(n) + pnm_s + self.net_s
    return max(0.0, t + service + margin - out)

  def plan(self, batch, now, predictor, pass_s, pnm_s, batch_max):
    """perf time to dispatch at: now, or when the chosen late criticals are usable.
    pass_s(n) is the GPU time of a pass of n."""
    margin = 2.0 * predictor.jitter

    def starve(cid, t, n):
      return self._starve(cid, t, n, pass_s, pnm_s, margin, predictor)

    have = {r.client_id for r in batch}
    late = sorted((max(t, now) + self.unpickle_s + predictor.jitter, c)
                  for c, t in predictor.upcoming(now, pass_s(len(batch))) if c not in have)
    best = None
    for k in range(0, min(len(late), batch_max - len(batch)) + 1):
      t = now if k == 0 else late[k - 1][0]
      n = len(batch) + k
      rest = min(len(late) - k, batch_max)
      cost = sum(starve(r.client_id, t, n) for r in batch)
      cost += sum(starve(c, t, n) for _, c in late[:k])
      cost += sum(starve(c, max(u, t + pass_s(n)), rest) for u, c in late[k:])
      key = (round(cost, 4), -k, t)
      if best is None or key < best[0]:
        best = (key, t)
    return best[1]

  def steal_room(self, batch, now, predictor, pass_s, pnm_s, room):
    """Unpadded: how many requests (at most room) a critical pass dispatched now may
    steal while every critical in it still gets its reply by its run-out. Nothing
    if any critical's run-out is unknown (its first request)."""
    if any(self._runout(r.client_id, predictor) is None for r in batch):
      return 0
    margin = 2.0 * predictor.jitter
    for s in range(room, 0, -1):
      if all(self._starve(r.client_id, now, len(batch) + s, pass_s, pnm_s, margin, predictor) == 0.0
             for r in batch):
        return s
    return 0


_EMPTY = object()


def _poll(q, timeout=0.0):
  """Next item of q, or _EMPTY if none came within timeout (None is the sentinel)."""
  try:
    return q.get(timeout=timeout) if timeout > 0 else q.get_nowait()
  except queue.Empty:
    return _EMPTY


def critical_worker(crit_queue, model, device, results, results_lock, t0_box,
                    expected_requests, done_event, warmup_obs_cpu, warmup_barrier,
                    autocast_dtype=None, crit_batch_max=1, crit_gather_s=0.0,
                    predictor=None, gpu_model=None, stream_priority=0, guard=None,
                    request_queue=None, offload=None, router=None, held=None, pad=True):
  """Dedicated worker for the critical queue, on its own CUDA stream.

  With crit_batch_max == 1 it serves strict batch-1. Above that it coalesces
  criticals: a robot's first request waits the whole crit_gather_s for the
  robots starting with it; after that the CriticalPlanner decides whether to wait
  for criticals predicted to land during the next pass. Without a predictor
  ('prio-streams') it lingers the fixed crit_gather_s. With pad the pass is padded
  to crit_batch_max and spare slots are filled from the normal queue; without, it
  runs at its own size and steals only what the criticals' run-outs allow.

  At the end of each pass it records when those robots are due back, which is
  what the normal queue's gate schedules around."""
  cudagraph_thread_init()
  stream, worker_id = get_stream_context(device, priority=stream_priority)
  try:
    with cudagraph_record_lock:
      for _ in range(2):
        if crit_batch_max > 1:
          run_batch_stage(model, [warmup_obs_cpu] * crit_batch_max, device, stream,
                          autocast_dtype, pad_to=crit_batch_max, offload=offload)
        else:
          run_stage(model, warmup_obs_cpu, device, stream, autocast_dtype, offload=offload)
  except BaseException:
    warmup_barrier.abort()   # else serve() blocks here until the ready-timeout
    raise
  warmup_barrier.wait()

  planner = CriticalPlanner()
  pnm_s = offload.denoise_s if offload is not None else 0.0

  def pass_s(n):
    return gpu_model.cost("critical", n, 0.2) if gpu_model is not None else 0.2

  while True:
    first = crit_queue.get()
    if first is None:
      crit_queue.task_done()
      break

    batch = [first]
    stop_after = False
    # Gather: wait for criticals predicted to land during the next pass only if
    # that causes less starvation than serving them later (CriticalPlanner).
    now = time.perf_counter()
    gather_deadline = now
    if crit_batch_max > 1 and crit_gather_s > 0:
      if predictor is not None:
        if first.obs.get("last_recv_epoch") is None:
          # A robot's first request has no prediction behind it, nor do robots
          # starting with it, so wait the whole window for them.
          gather_deadline = now + crit_gather_s
        else:
          while len(batch) < crit_batch_max:     # everything already waiting
            r = _poll(crit_queue)
            if r is _EMPTY:
              break
            if r is None:
              stop_after = True
              break
            batch.append(r)
          gather_deadline = planner.plan(batch, now, predictor, pass_s, pnm_s, crit_batch_max)
      else:
        # No predictor ('prio-streams' baseline): fall back to a blind linger,
        # i.e. the stricter fixed deadline that arm's critical stream is defined
        # by. A lone critical therefore pays the full window here, which is
        # exactly what the predictive path above exists to avoid.
        gather_deadline = now + crit_gather_s
    while len(batch) < crit_batch_max and not stop_after:
      remaining = gather_deadline - time.perf_counter()
      try:
        req = crit_queue.get(timeout=remaining) if remaining > 0 else crit_queue.get_nowait()
      except queue.Empty:
        break
      if req is None:  # sentinel mid-gather: serve what we have, then stop
        crit_queue.task_done()
        stop_after = True
        break
      batch.append(req)

    n_crit = len(batch)
    # WORK STEAL: padded, the pass runs at crit_batch_max regardless, so the unused
    # slots run duplicated observations whose results are discarded. Filling them
    # from the normal queue costs zero extra GPU and does not delay the critical
    # by one microsecond -- same shape, same kernels, same duration. Language is
    # per-request (tokenized client-side), so mixing tasks in one batch is exact.
    # Unpadded, every stolen request lengthens the pass, so steal only as many as
    # still get each critical its reply by its run-out (CriticalPlanner.steal_room).
    if crit_batch_max > 1 and request_queue is not None:
      room = crit_batch_max - len(batch)
      if not pad:
        room = planner.steal_room(batch, time.perf_counter(), predictor, pass_s,
                                  pnm_s, room)
      want = len(batch) + room
      if held is not None:
        batch.extend(held.take(want - len(batch)))
      while len(batch) < want:
        try:
          stolen = request_queue.get_nowait()
        except queue.Empty:
          break
        if stolen is None:            # shutdown sentinel: not ours to consume
          request_queue.task_done()   # balance this get before re-queueing it
          request_queue.put(None)
          break
        batch.append(stolen)

    if router is not None:
      router.set([r.obs["expert_id"] for r in batch])
    worker_start = time.perf_counter()
    dispatch_epoch = time.time()
    for req in batch:
      dispatch_obs_epoch(req, dispatch_epoch)
    for req in batch[:n_crit]:
      planner.observe(req)
    with (guard.critical_pass() if guard is not None else contextlib.nullcontext()):
      if crit_batch_max > 1:
        chunks, gpu_ms = run_batch_stage(model, [r.obs for r in batch], device,
                                         stream, autocast_dtype,
                                         pad_to=crit_batch_max if pad else None,
                                         offload=offload)
      else:
        chunk, gpu_ms = run_stage(model, batch[0].obs, device, stream, autocast_dtype,
                                  offload=offload)
        chunks = [chunk]
      worker_end = time.perf_counter()
      # The GPU is free at worker_end. Under offload the action expert still has to
      # run on the PNM units, so the reply lands later -- but this worker does NOT wait
      # for it, which is the entire point: the next batch starts now.
      ready = reply_time(worker_end, offload)
      # Predict each critical's return BEFORE the guard lets a normal batch start;
      # the handler sends the reply (and refines this) only after that batch looked.
      if predictor is not None:
        for req in batch:
          if req.priority:
            predictor.on_reply(req.client_id, ready, req.obs)

    for i, req in enumerate(batch):
      req.result = pickle.dumps((chunks[i], req.obs["obs_epoch"]),
                                protocol=pickle.HIGHEST_PROTOCOL)
      req.ready_at = ready
      req.event.set()
      (crit_queue if i < n_crit else request_queue).task_done()
    if gpu_model is not None:
      # Learn the WHOLE pass (H2D + kernels + sync), in seconds: that is what a
      # critical actually waits behind, not kernel time alone.
      gpu_model.learn("critical", len(batch), worker_end - worker_start)
    if guard is not None:
      guard.critical_served(n_crit)

    t0 = t0_box[0]
    inference_wall_ms = (worker_end - worker_start) * 1000.0
    with results_lock:
      results.extend({
        "client_id": req.client_id,
        "step": req.step,
        "worker_id": worker_id,
        "arrival_time_s": req.recv_time - t0,
        "queue_wait_ms": (worker_start - req.enqueue_time) * 1000.0,
        "gpu_time_ms": gpu_ms,
        "inference_wall_ms": inference_wall_ms,
        "server_total_ms": (ready - req.recv_time) * 1000.0,
        "completion_time_s": ready - t0,
        "action_steps": int(chunks[i].shape[0]),
        "batch_size": len(batch),
        "priority": req.priority,
        # 'fill' = a normal request that rode along in a padding slot for free.
        "queue": "critical" if i < n_crit else "fill",
        "gate_held_ms": 0.0,
        "gate_slack_ms": 0.0,
        "offload_ms": (ready - worker_end) * 1000.0,
        "expert_id": req.obs["expert_id"],
      } for i, req in enumerate(batch))
      n_done = len(results)

    print(f"[server] CRIT batch={n_crit}+{len(batch) - n_crit}fill"
          f" | gpu {gpu_ms:7.1f} ms | wall {inference_wall_ms:7.1f} ms | served {n_done}")

    if expected_requests and n_done >= expected_requests:
      done_event.set()
    if stop_after:
      break


def batching_worker(request_queue, model, device, results, results_lock, t0_box,
                    expected_requests, done_event, warmup_obs_cpu, warmup_barrier,
                    autocast_dtype, batch_max, batch_timeout_s, gate=None,
                    gpu_model=None, guard=None, offload=None, router=None, held=None,
                    pad=True):
  """One dynamic batcher, on its own CUDA stream. It opens a batch on the first
  queued request, then keeps pulling until it has batch_max requests OR the
  deadline (batch_timeout_s from the first request) elapses -- whichever comes
  first -- then issues that (possibly partial) batch as one forward pass. Every
  policy that has a normal queue runs exactly one of these.

  Every forward pass runs at the fixed shape batch_max unless pad=False (partial
  batches are padded in run_batch_inference), so torch.compile stays STATIC -- one specialized, fast
  graph, avoiding the ~2x-slower dynamic graph. Under plain 'batching' this
  worker's warmup is the only compile."""
  cudagraph_thread_init()
  stream, worker_id = get_stream_context(device)
  try:
    with cudagraph_record_lock:
      for _ in range(3):
        run_batch_stage(model, [warmup_obs_cpu] * batch_max, device, stream,
                        autocast_dtype, pad_to=batch_max, offload=offload)
  except BaseException:
    warmup_barrier.abort()   # else serve() blocks here until the ready-timeout
    raise
  warmup_barrier.wait()

  held = held if held is not None else HeldBatch()
  stop_after = False
  while True:
    # What the last admission left held opens this batch; otherwise wait for one.
    if held.size() == 0:
      if stop_after:
        break
      first = request_queue.get()
      if first is None:  # shutdown sentinel
        request_queue.task_done()
        break
      held.add(first)

    # Gather up to batch_max within the deadline measured from the first request,
    # into the shared held batch so critical passes can steal from it meanwhile.
    deadline = time.perf_counter() + batch_timeout_s
    while held.size() < batch_max and not stop_after:
      remaining = deadline - time.perf_counter()
      if remaining <= 0:
        break
      try:
        req = request_queue.get(timeout=remaining)
      except queue.Empty:
        break  # deadline hit -> fire the partial batch
      if req is None:  # sentinel mid-gather: serve what we have, then stop
        request_queue.task_done()
        stop_after = True
        break
      held.add(req)

    # Admission control: hold the batch if it would still be on the GPU when the
    # next critical is predicted to land (the GPU will not preempt it).
    # The gate and the guard must both allow the pass at the same instant. A
    # guard wait makes the gate's decision stale -- the criticals it waited for
    # have been answered and are now predicted back -- so after any guard wait the
    # gate decides again. Both together are capped at the guard's starve_cap_s.
    held_ms, slack_ms, n = 0.0, 0.0, None
    admit_deadline = time.perf_counter() + (guard.starve_cap_s if guard is not None else 0.0)
    while True:
      if gate is not None:
        h, slack_ms, n = gate.wait_until_safe(held)
        held_ms += h
      if guard is None or not guard.wait_clear(admit_deadline):
        break
      if gate is None or time.perf_counter() >= admit_deadline:
        break
    # The cap ends the wait for predicted criticals, never for one already queued or
    # on the GPU: it clears within one critical pass, and starting now would share it.
    if guard is not None:
      guard.wait_clear(time.perf_counter() + guard.starve_cap_s)
    batch = held.close(n)
    if not batch:                 # every request rode a critical pass meanwhile
      continue
    if router is not None:
      router.set([r.obs["expert_id"] for r in batch])

    # Start the clock AFTER admission: blocking is time the GPU was not running
    # this batch, and charging it to B makes the gate defer more, which causes
    # more blocking -- a feedback loop.
    worker_start = time.perf_counter()
    dispatch_epoch = time.time()
    chunks, gpu_ms = run_batch_stage(model, [r.obs for r in batch], device,
                                     stream, autocast_dtype,
                                     pad_to=batch_max if pad else None, offload=offload)
    worker_end = time.perf_counter()
    if gpu_model is not None:
      gpu_model.learn("normal", len(batch), worker_end - worker_start)   # the pass only, seconds
    # B stays GPU-only on purpose: the gate schedules around GPU occupancy, and
    # under offload this batch stops occupying the GPU at worker_end even though
    # its reply is still pending on the PNM units.
    ready = reply_time(worker_end, offload)

    for i, req in enumerate(batch):
      req.result = pickle.dumps((chunks[i], dispatch_obs_epoch(req, dispatch_epoch)),
                                protocol=pickle.HIGHEST_PROTOCOL)
      req.ready_at = ready
      req.event.set()  # wake the parked gRPC handler
      request_queue.task_done()

    t0 = t0_box[0]
    bs = len(batch)
    inference_wall_ms = (worker_end - worker_start) * 1000.0
    batch_records = [{
      "client_id": req.client_id,
      "step": req.step,
      "worker_id": worker_id,
      "arrival_time_s": req.recv_time - t0,
      # queue wait includes the batch-gather/deadline wait for the request(s)
      # that opened the batch; near-zero for the last one to arrive.
      "queue_wait_ms": (worker_start - req.enqueue_time) * 1000.0,
      "gpu_time_ms": gpu_ms,               # one batched forward pass (shared)
      "inference_wall_ms": inference_wall_ms,
      "server_total_ms": (ready - req.recv_time) * 1000.0,
      "completion_time_s": ready - t0,
      "action_steps": int(chunks[i].shape[0]),
      "batch_size": bs,
      "priority": req.priority,
      "queue": "normal",
      "gate_held_ms": held_ms,   # time this batch was held for a critical
      "gate_slack_ms": slack_ms, # headroom to the predicted critical arrival
      "offload_ms": (ready - worker_end) * 1000.0,
      "expert_id": req.obs["expert_id"],
    } for i, req in enumerate(batch)]
    with results_lock:
      results.extend(batch_records)
      n_done = len(results)

    print(f"[server] batch={bs}/{batch_max} | gpu {gpu_ms:7.1f} ms | "
          f"wall {inference_wall_ms:7.1f} ms | held {held_ms:6.1f} ms | served {n_done}")

    if expected_requests and n_done >= expected_requests:
      done_event.set()


# ---------------------------------------------------------------------------
# Armory baseline (Bansal/He et al., "Action Chunk Scheduling for Batched Robot
# Policy Serving"; github.com/GaTech-RL2/armory at e876202). Only its scheduling DECISION is ported:
# which waiting robots to serve, and when. It shares none of our mechanisms (no
# critical queue, gate, guard, stealing or arrival prediction). Batches are
# padded to batch_max, as in our arms. A request stands in for Armory's per-tick
# observation stream (whose contents do not affect timing): the robot sends it at
# its request point and the scheduler decides when to serve it. Departures
# from their code, both in Armory's favour or forced by our client:
#   - every candidate batch is scored; theirs stops after 8 when the GPU is
#     free, which never reaches batches above 3 once both classes wait;
#   - network delays are measured from two timestamps each request carries,
#     instead of Armory's warmup pings and per-chunk acks.
# ---------------------------------------------------------------------------

ARMORY_HORIZON_S = 1.0   # lookahead_actions.HORIZON


class BatchLatency:
  """GPU time per batch size, in seconds: an EWMA at alpha 0.1, seeded from
  warmup (armory EMALatencyTracker.update_infer + the GPU's BatchProfile)."""

  def __init__(self, seed, alpha=0.1):
    self._s = dict(seed)
    self.alpha = alpha

  def update(self, b, s):
    prev = self._s.get(b)
    self._s[b] = s if prev is None else (1 - self.alpha) * prev + self.alpha * s

  def get(self, b):
    return self._s[b]


class ArmoryMirror:
  """What each robot has left to play (Armory's Mirror, for our robots), plus its
  per-robot network delays (Armory's LatencyTracker: EWMA at alpha 0.1).

  A chunk dispatched at t lands at t + GPU time + action delay. It was made from
  the observation taken obs delay before dispatch, so the robot skips what it
  played since then, plays the rest up to h_max at control_hz, and stops.
  Deadline = when it stops. A robot with no chunk yet has nothing to play."""

  def __init__(self, alpha=0.1):
    self.alpha = alpha
    self._state = {}      # client_id -> (t_arrive, actions playable from that chunk)
    self._obs_lat = {}    # client_id -> s, request stamped -> server arrival
    self._act_lat = {}    # client_id -> s, reply sent -> robot received
    self._last = {}       # client_id -> the last request served (for its sent_epoch)

  def _ewma(self, d, key, v):
    v = max(0.0, v)
    d[key] = v if key not in d else (1 - self.alpha) * d[key] + self.alpha * v

  def observe(self, req):
    """A request entered the pool: learn this robot's delays from its stamps."""
    cid = req.client_id
    if req.obs.get("send_epoch") is not None:
      self._ewma(self._obs_lat, cid, req.recv_epoch - req.obs["send_epoch"])
    prev = self._last.get(cid)
    if prev is not None and prev.sent_epoch and req.obs.get("last_recv_epoch") is not None:
      self._ewma(self._act_lat, cid, req.obs["last_recv_epoch"] - prev.sent_epoch)

  def obs_lat(self, req):
    return self._obs_lat.get(req.client_id, 0.0)

  def act_lat(self, req):
    return self._act_lat.get(req.client_id, 0.0)

  @staticmethod
  def _hz(req):
    return float(req.obs["control_hz"])

  def remaining(self, req, t):
    """Actions still to play at time t (fractional: the robot's clock is known)."""
    st = self._state.get(req.client_id)
    if st is None:
      return 0.0
    t_arrive, playable = st
    return max(0.0, playable - self._hz(req) * max(0.0, t - t_arrive))

  def _playable(self, req, t_dispatch, d):
    """Actions a chunk dispatched at t_dispatch, taking d on the GPU, gives the
    robot, and when it lands."""
    t_obs = t_dispatch - self.obs_lat(req)
    t_arrive = t_dispatch + d + self.act_lat(req)
    skipped = min(self.remaining(req, t_obs), self._hz(req) * (t_arrive - t_obs))
    return max(0.0, req.obs["h_max"] - skipped), t_arrive

  def on_reply(self, req, t_dispatch, t_end):
    playable, t_arrive = self._playable(req, t_dispatch, t_end - t_dispatch)
    self._state[req.client_id] = (t_arrive, playable)
    self._last[req.client_id] = req

  def deadline(self, req):
    st = self._state.get(req.client_id)
    if st is None:
      return req.recv_time
    t_arrive, playable = st
    return t_arrive + playable / self._hz(req)

  def gain(self, req, now, d):
    """Seconds of action a chunk dispatched now delivers within Armory's horizon."""
    playable, t_arrive = self._playable(req, now, d)
    return min(playable / self._hz(req), max(0.0, now + ARMORY_HORIZON_S - t_arrive))


def _tier_prefix_batches(tiers, batch_max):
  """Every batch made of a prefix of each tier (armory _candidate_batches)."""
  def compositions(n, caps):
    if len(caps) == 1:
      if n <= caps[0]:
        yield (n,)
      return
    for i in range(min(n, caps[0]) + 1):
      for rest in compositions(n - i, caps[1:]):
        yield (i,) + rest

  pool = sum(len(t) for t in tiers)
  for size in range(1, min(pool, batch_max) + 1):
    for counts in compositions(size, [len(t) for t in tiers]):
      yield [r for t, c in zip(tiers, counts) for r in t[:c]]


def lookahead_batch(pending, mirror, latency, batch_max, crit_weight, now):
  """Armory Lookahead at its defaults: depth 1, no idle actions, 1 s horizon.

  Candidates: waiting robots sorted by deadline, grouped by weight (highest
  first), prefixes of each group. Score: weighted seconds of action the batch's
  own chunks deliver inside the horizon, per second of GPU time. Only served
  robots score -- Armory counts actions from the searched chunks alone -- and a
  robot still playing its old chunk skips part of the new one, so it gains less."""
  weight = lambda r: crit_weight if r.priority else 1.0
  tiers = {}
  for r in sorted(pending, key=mirror.deadline):
    tiers.setdefault(weight(r), []).append(r)
  tiers = [tiers[w] for w in sorted(tiers, reverse=True)]

  best, best_obj = None, float("-inf")
  for batch in _tier_prefix_batches(tiers, batch_max):
    d = latency.get(len(batch))
    obj = sum(weight(r) * mirror.gain(r, now, d) for r in batch) / d
    if obj > best_obj:
      best, best_obj = batch, obj
  return best


def edf_batch(pending, mirror, latency, batch_max, now):
  """Armory GreedyDeadlineScheduler: earliest deadline first, sized to the
  largest batch that still finishes before the earliest deadline, else the most
  efficient batch size. The deadline is taken net of the robot's reply delay."""
  infer_deadline = lambda r: mirror.deadline(r) - mirror.act_lat(r)
  ordered = sorted(pending, key=infer_deadline)
  slack = infer_deadline(ordered[0]) - now
  fits = [b for b in range(batch_max, 0, -1) if latency.get(b) <= slack]
  b = fits[0] if fits else max(range(1, batch_max + 1), key=lambda b: b / latency.get(b))
  return ordered[:b]


def armory_worker(request_queue, model, device, results, results_lock, t0_box,
                  expected_requests, done_event, warmup_barrier, autocast_dtype,
                  batch_max, latency, mirror, policy, crit_weight, router, offload=None,
                  pad=True):
  """One worker, one stream, one pool of waiting requests. Whenever the GPU is
  free and anything is waiting, the policy picks a subset and the GPU runs
  exactly that subset. Work-conserving, like Armory at max_in_flight=1 with no
  idle actions. Every shape was compiled on the main thread before this starts.

  Every pass is padded to batch_max unless pad=False, which runs each batch at its
  own size as Armory does. With an offload (armory-pnm) the GPU runs the prefill
  only and the PNM units take any batch up to 5 at once, so no batch queues between
  the two; the GPU is free for the next batch as soon as the prefill ends."""
  cudagraph_thread_init()
  stream, worker_id = get_stream_context(device)
  warmup_barrier.wait()

  pending, stop = [], False
  while True:
    if not pending and not stop:
      req = request_queue.get()
      if req is None:
        stop = True
      else:
        mirror.observe(req)
        pending.append(req)
      request_queue.task_done()
    while not stop:
      try:
        req = request_queue.get_nowait()
      except queue.Empty:
        break
      if req is None:
        stop = True
      else:
        mirror.observe(req)
        pending.append(req)
      request_queue.task_done()
    if not pending:
      break

    now = time.perf_counter()
    batch = (lookahead_batch(pending, mirror, latency, batch_max, crit_weight, now)
             if policy in ("armory", "armory-pnm")
             else edf_batch(pending, mirror, latency, batch_max, now))
    chosen = {id(r) for r in batch}
    pending = [r for r in pending if id(r) not in chosen]

    router.set([r.obs["expert_id"] for r in batch])
    worker_start = time.perf_counter()
    dispatch_epoch = time.time()
    # Padded to batch_max unless pad=False (then each batch at its own size).
    chunks, gpu_ms = run_batch_stage(model, [r.obs for r in batch], device, stream,
                                     autocast_dtype, pad_to=batch_max if pad else None,
                                     offload=offload)
    worker_end = time.perf_counter()
    ready = reply_time(worker_end, offload)
    latency.update(len(batch), ready - worker_start)

    for i, req in enumerate(batch):
      # The robot skips what it played after this chunk's observation was taken.
      obs_epoch = dispatch_epoch - mirror.obs_lat(req)
      req.result = pickle.dumps((chunks[i], obs_epoch), protocol=pickle.HIGHEST_PROTOCOL)
      req.ready_at = ready
      mirror.on_reply(req, worker_start, ready)
      req.event.set()

    t0 = t0_box[0]
    inference_wall_ms = (worker_end - worker_start) * 1000.0
    with results_lock:
      results.extend({
        "client_id": req.client_id,
        "step": req.step,
        "worker_id": worker_id,
        "arrival_time_s": req.recv_time - t0,
        "queue_wait_ms": (worker_start - req.enqueue_time) * 1000.0,
        "gpu_time_ms": gpu_ms,
        "inference_wall_ms": inference_wall_ms,
        "server_total_ms": (ready - req.recv_time) * 1000.0,
        "completion_time_s": ready - t0,
        "action_steps": int(chunks[i].shape[0]),
        "batch_size": len(batch),
        "priority": req.priority,
        "queue": policy,
        "gate_held_ms": 0.0,
        "gate_slack_ms": 0.0,
        "offload_ms": (ready - worker_end) * 1000.0,
        "expert_id": req.obs["expert_id"],
      } for i, req in enumerate(batch))
      n_done = len(results)

    n_crit = sum(r.priority for r in batch)
    print(f"[server] {policy} batch={len(batch)} ({n_crit} crit) waiting={len(pending)} | "
          f"gpu {gpu_ms:7.1f} ms | wall {inference_wall_ms:7.1f} ms | served {n_done}")

    if expected_requests and n_done >= expected_requests:
      done_event.set()
