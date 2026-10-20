"""GPU execution paths for the policy server.

Everything that touches the model or the device lives here: building the policy,
moving observations onto it, and the forward passes themselves -- batch-1, padded
batch, and the prefill-only pass used by the 'aegis-pnm' offload arm. The schedulers in
scheduling.py decide WHAT to run and WHEN; this module only knows HOW to run it.

run_batch_stage()/run_stage() are the seam between the two: both offload arms go
through one function so the scheduler around them is provably identical.
"""

import contextlib
import itertools
import os
import threading
import time
from collections import defaultdict

import torch

from robot_client import (
  BENCHMARKS,
  build_features,
  build_observation_batch,
  build_preprocessor,
)

# Model classes (the server holds the actual weights; the client only preprocesses).
from lerobot.configs.types import NormalizationMode
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


# Per-worker CUDA stream + stable worker id, assigned lazily on first use. One
# stream per worker thread is what lets the critical worker's pass overlap with
# the normal batcher's instead of serializing behind it on the default stream.
_thread_state = threading.local()
_worker_counter = itertools.count()
_worker_lock = threading.Lock()


def get_stream_context(device, priority=0):
  """Return this worker thread's (stream, worker_id), creating them on first
  call. On CPU there is no stream (returns None).

  priority is the CUDA stream priority (lower = higher priority; -1 is the top
  band on sm70). It is NOT preemption: blocks already resident on the SMs run to
  completion, so it only decides which PENDING blocks get dispatched next. That
  still helps a critical request that arrives mid-batch, because a normal batch
  is many kernel launches long."""
  if not hasattr(_thread_state, "worker_id"):
    with _worker_lock:
      _thread_state.worker_id = next(_worker_counter)
    _thread_state.stream = (torch.cuda.Stream(device=device, priority=priority)
                            if device.type == "cuda" else None)
  return _thread_state.stream, _thread_state.worker_id


DEFAULT_COMPILE_MODE = "max-autotune-no-cudagraphs"

# Capture is exclusive process-wide even though replay is not: two workers
# recording at once fail with cudaErrorStreamCaptureUnsupported. Held only around
# each worker's warmup, which is where recording happens.
cudagraph_record_lock = threading.Lock()


def cudagraph_thread_init():
  """Give this worker thread its own cudagraph tree manager.

  torch keeps the manager in a threading.local populated at import time, so a
  thread that did not import it asserts inside get_container. Private API, hence
  the guard -- if it ever moves, say so rather than failing inside a worker."""
  try:
    from torch._inductor import cudagraph_trees as ct
    if not hasattr(ct.local, "tree_manager_containers"):
      ct.local.tree_manager_containers = {}
      ct.local.tree_manager_locks = defaultdict(threading.Lock)
  except Exception as e:  # noqa: BLE001
    print(f"[server] cudagraph thread init failed ({e}); "
          f"run with --compile-mode max-autotune-no-cudagraphs")


def use_library_cumsum():
  """Compile torch.cumsum as a call to ATen's kernel instead of a generated Triton scan.

  Inductor splits a long scan across thread blocks when the GPU has enough SMs for it
  (pi0's position ids, B x 560, on the 84-SM RTX A6000), and that split-scan kernel
  faults with an illegal memory access when its compiled graph is reloaded from the
  cache. The library kernel avoids it; cumsum is a negligible part of a pass."""
  from torch._inductor import lowering

  def fallback(x, axis=None, dtype=None):
    return lowering.fallback_cumsum(x, dim=axis, dtype=dtype)

  for op in (torch.ops.aten.cumsum, torch.ops.aten.cumsum.default):
    lowering.lowerings[op] = fallback


def setup_model(model_choice, benchmark, device,
                compile_model=False, compile_mode="max-autotune-no-cudagraphs",
                precision="fp16"):
  """Build a model for the chosen benchmark spec. Random-init weights give the
  same latency as real ones AT A FIXED DTYPE -- but dtype is decisive: on a V100
  (no TF32/tensor cores for fp32) fp32 matmuls can't be accelerated by
  torch.compile, while fp16 hits tensor cores for a ~5x speedup. So precision
  matters as much as compile.

  precision="fp16" builds the realistic half-precision model matching
  lerobot/smolvla_base (~68ms compiled here); "fp32" is the slower reference.
  For smolvla, fp16 means load_vlm_weights=True (lerobot loads the VLM as
  torch_dtype=float16; this downloads SmolVLM2-500M once, then caches). fp32
  means random weights -- but lerobot builds those from the hub AutoConfig,
  whose torch_dtype is BFLOAT16, so the fp32 path must cast explicitly (see
  below) or it silently runs bf16, which sm70 has no tensor cores for and
  emulates via magma_sgemmEx at ~3x the true fp32 cost. For pi0, weights are
  always fp32 and precision only toggles the inference-time autocast.

  compile_model wraps the inner flow-matching model's sample_actions() (the
  function predict_action_chunk() calls) in torch.compile. compile_mode is
  applied to both models -- note pi0's own default is plain "max-autotune", so
  we override it here."""
  spec = BENCHMARKS[benchmark]
  input_features, output_features = build_features(spec)
  fp16 = precision == "fp16"
  if compile_model and os.environ.get("VLA_LIBRARY_CUMSUM") == "1":
    use_library_cumsum()
    print("  cumsum compiled as a library call (VLA_LIBRARY_CUMSUM=1)")

  # IDENTITY normalization everywhere -> no dataset_stats needed.
  norm = {
    "VISUAL": NormalizationMode.IDENTITY,
    "STATE": NormalizationMode.IDENTITY,
    "ACTION": NormalizationMode.IDENTITY,
  }

  print(f"Building '{model_choice}' for benchmark '{benchmark}' on {device} (precision={precision})...")
  print(f"  cameras={list(spec['cameras'])} state_dim={spec['state_dim']} action_dim={spec['action_dim']}")
  print(f"  torch.compile={compile_model} (mode={compile_mode})" if compile_model
        else "  torch.compile=off")

  if model_choice == "pi0":
    # pi0's config accepts float32/bfloat16, never float16. "fp16" therefore means
    # fp32 weights plus a float16 autocast at inference (set up in serve() ->
    # run_inference), which is the only tensor-core path on a V100. "bf16" builds
    # real bfloat16 weights instead and needs no autocast -- sm80 and later only,
    # and it halves every weight read, which matters most for the gathered
    # per-task experts.
    if fp16:
      print("  pi0 fp16 = fp32 weights + float16 autocast at inference")
    config = PI0Config(
      input_features=input_features,
      output_features=output_features,
      image_resolution=(224, 224),  # pi0 requires a square resize target
      device=str(device),
      dtype="bfloat16" if precision == "bf16" else "float32",
      normalization_mapping=norm,
      compile_model=compile_model,
      compile_mode=compile_mode,
    )
    model = PI0Policy(config)
  elif model_choice == "smolvla":
    config = SmolVLAConfig(
      input_features=input_features,
      output_features=output_features,
      device=str(device),
      # fp16/bf16 -> load the real VLM (smolvlm_with_expert pins torch_dtype
      # float16; bf16 is reached by casting below). fp32 -> random fp32 weights,
      # which skips the download but is the slow reference.
      load_vlm_weights=precision in ("fp16", "bf16"),
      normalization_mapping=norm,
      pad_language_to="max_length",
      compile_model=compile_model,
      compile_mode=compile_mode,
    )
    model = SmolVLAPolicy(config)
    if precision == "fp32":
      # load_vlm_weights=False builds the VLM from the hub AutoConfig, whose dtype
      # is not fp32 -> cast so the fp32 reference really is fp32.
      model.to(dtype=torch.float32)
    elif precision == "bf16":
      model.to(dtype=torch.bfloat16)
  else:
    raise ValueError(f"Unknown model choice: {model_choice}")

  model.to(device)
  model.eval()
  return model, spec


def move_to_device(obs, device):
  """Move the tensor values of a (client-preprocessed, CPU) observation onto the
  server's device; leave non-tensor values (task string, etc.) untouched."""
  return {
    key: (value.to(device) if isinstance(value, torch.Tensor) else value)
    for key, value in obs.items()
  }


def run_inference(model, obs_cpu, device, stream, autocast_dtype=None):
  """Move a CPU observation to the device and run predict_action_chunk on the
  given stream. Returns (chunk_cpu of shape (n_action_steps, action_dim),
  gpu_ms), where gpu_ms is CUDA-event device time (H2D + compute) on cuda, else
  wall time. This is the batch-1 path, reached via run_stage() when the critical
  worker runs with crit_batch_max == 1; warmup goes through the same function so
  both hit identical torch.compile guards.

  autocast_dtype (e.g. torch.float16) enables mixed-precision autocast around the
  forward pass: pi0's "fp16" keeps fp32 weights (its config rejects float16) and
  autocast runs the matmuls in fp16."""
  amp = (torch.autocast(device_type="cuda", dtype=autocast_dtype)
         if autocast_dtype is not None else contextlib.nullcontext())
  if device.type == "cuda":
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode(), torch.cuda.stream(stream):
      start_event.record(stream)
      obs = move_to_device(obs_cpu, device)
      with amp:
        actions = model.predict_action_chunk(obs)  # (1, n_action_steps, action_dim)
      chunk_gpu = actions[0]  # -> (n_action_steps, action_dim) for this client
      end_event.record(stream)
      stream.synchronize()
    return chunk_gpu.detach().cpu(), start_event.elapsed_time(end_event)

  dev_start = time.perf_counter()
  with torch.inference_mode():
    obs = move_to_device(obs_cpu, device)
    with amp:
      actions = model.predict_action_chunk(obs)
    chunk = actions[0].detach().cpu()
  return chunk, (time.perf_counter() - dev_start) * 1000.0


def collate_obs(obs_cpu_list):
  out = {}
  for k, v0 in obs_cpu_list[0].items():
    if torch.is_tensor(v0):
      out[k] = torch.cat([o[k] for o in obs_cpu_list], dim=0)
    else:
      out[k] = v0
  return out


def run_batch_inference(model, obs_cpu_list, device, stream, autocast_dtype=None, pad_to=None):
  """Run ONE batched forward pass over a list of B observations. Returns
  (list of B chunk_cpu tensors, gpu_ms). This is the batching serving path: the
  same kernels amortize over B requests instead of B separate batch-1 passes.

  When pad_to is set and B < pad_to, the batch is padded up to pad_to (repeating
  the first observation) so the forward pass always runs at ONE fixed shape.
  That keeps torch.compile STATIC (a single specialized, fast graph) instead of
  the ~2x-slower dynamic graph you get from feeding it varying batch sizes. Only
  the first B (real) outputs are returned; the padded rows are discarded. gpu_ms
  therefore reflects the full padded-batch cost -- the real cost we always pay."""
  real = len(obs_cpu_list)
  batched_cpu = collate_obs(obs_cpu_list)
  if pad_to is not None and real < pad_to:
    pad_n = pad_to - real
    for k, v in list(batched_cpu.items()):
      if torch.is_tensor(v):
        reps = v[0:1].repeat(pad_n, *([1] * (v.dim() - 1)))
        batched_cpu[k] = torch.cat([v, reps], dim=0)
  amp = (torch.autocast(device_type="cuda", dtype=autocast_dtype)
         if autocast_dtype is not None else contextlib.nullcontext())
  if device.type == "cuda":
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode(), torch.cuda.stream(stream):
      start_event.record(stream)
      obs = move_to_device(batched_cpu, device)
      with amp:
        actions = model.predict_action_chunk(obs)  # (pad_to, n_action_steps, action_dim)
      end_event.record(stream)
      stream.synchronize()
    chunks = [actions[i].detach().cpu() for i in range(real)]  # drop padded rows
    return chunks, start_event.elapsed_time(end_event)

  dev_start = time.perf_counter()
  with torch.inference_mode():
    obs = move_to_device(batched_cpu, device)
    with amp:
      actions = model.predict_action_chunk(obs)
    chunks = [actions[i].detach().cpu() for i in range(real)]
  return chunks, (time.perf_counter() - dev_start) * 1000.0


# Simulated PNM latency of the action expert, in ms (all 10 denoising steps). The GPU
# has 2 HBM2e stacks (the A6000's bandwidth); each denoises one request, so a batch
# of up to PNM_STACKS takes the same time as one request, and no batch may be larger.
PNM_DENOISE_MS = {"smolvla": 29.0, "pi0": 38.6}
PNM_STACKS = 2


class PnmOffload:
  """Splits the policy across the GPU and processing-near-memory (PNM) units.

  The GPU runs ONLY the VLM prefix prefill and hands the KV cache off; the
  flow-matching denoising runs on the PNM units. The GPU is therefore released
  the instant prefill ends and can start the next batch while the previous one is
  still denoising -- the two stages pipeline instead of serializing.

  denoise_s is measured and deterministic, so a batch's reply time is known the
  moment prefill finishes. Nothing blocks a GPU worker to account for it: the
  worker stamps ready_at and moves on, and the parked gRPC handler holds the reply
  until then. Blocking the worker instead would defeat the entire architecture.

  Action VALUES are not simulated. Weights are random-init in this benchmark, so
  every chunk is already numerically meaningless; what has to be faithful is the
  shape of the reply and the instant it arrives."""

  def __init__(self, denoise_s, chunk_shape, chunk_dtype):
    self.denoise_s = denoise_s
    self._chunk = torch.zeros(chunk_shape, dtype=chunk_dtype)
    # The stacks take one batch at a time. A denoise is shorter than any prefill, so
    # a batch always finds them free unless two prefills overlapped on the GPU.
    self._lock = threading.Lock()
    self._free = 0.0

  def chunk(self):
    """A correctly-shaped stand-in for a denoised action chunk."""
    return self._chunk

  def reply_time(self, gpu_done):
    """When a batch that left the GPU at gpu_done may be answered."""
    with self._lock:
      self._free = max(gpu_done, self._free) + self.denoise_s
      return self._free


def run_batch_prefill(model, obs_cpu_list, device, stream, autocast_dtype=None, pad_to=None):
  """The GPU half of the offloaded policy: ONE batched VLM prefix prefill and
  nothing else. Returns gpu_ms (CUDA-event device time, H2D + prefill).

  Padding follows run_batch_inference exactly, so torch.compile still sees a
  single static shape and the two arms differ only in which stages run.

  The prefix KV cache is computed and dropped. It is deliberately NOT copied to
  the host: on processing-near-memory units the action expert reads it in
  place, so a D2H round trip would charge this architecture for a host link it
  does not have. The denoising that consumes the cache is accounted for by
  PNM_DENOISE_MS."""
  real = len(obs_cpu_list)
  batched_cpu = collate_obs(obs_cpu_list)
  if pad_to is not None and real < pad_to:
    pad_n = pad_to - real
    for k, v in list(batched_cpu.items()):
      if torch.is_tensor(v):
        reps = v[0:1].repeat(pad_n, *([1] * (v.dim() - 1)))
        batched_cpu[k] = torch.cat([v, reps], dim=0)
  amp = (torch.autocast(device_type="cuda", dtype=autocast_dtype)
         if autocast_dtype is not None else contextlib.nullcontext())
  if device.type == "cuda":
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode(), torch.cuda.stream(stream):
      start_event.record(stream)
      obs = move_to_device(batched_cpu, device)
      with amp:
        model.vlm_prefill(obs)
      end_event.record(stream)
      stream.synchronize()
    return start_event.elapsed_time(end_event)

  dev_start = time.perf_counter()
  with torch.inference_mode():
    obs = move_to_device(batched_cpu, device)
    with amp:
      model.vlm_prefill(obs)
  return (time.perf_counter() - dev_start) * 1000.0


def run_batch_stage(model, obs_cpu_list, device, stream, autocast_dtype=None, pad_to=None,
                    offload=None):
  """One GPU pass for a batch of requests.

  offload=None (every policy except 'aegis-pnm') runs the whole policy on the GPU and is
  exactly run_batch_inference. With an offload, the GPU runs prefill only and the
  returned chunks are placeholders; the action expert's cost is added off-GPU by
  the caller. Routing both arms through one function is deliberate -- the
  scheduler around it must be identical or the comparison is not controlled."""
  if offload is None:
    return run_batch_inference(model, obs_cpu_list, device, stream, autocast_dtype, pad_to)
  gpu_ms = run_batch_prefill(model, obs_cpu_list, device, stream, autocast_dtype, pad_to)
  return [offload.chunk()] * len(obs_cpu_list), gpu_ms


def run_stage(model, obs_cpu, device, stream, autocast_dtype=None, offload=None):
  """Batch-1 counterpart of run_batch_stage."""
  if offload is None:
    return run_inference(model, obs_cpu, device, stream, autocast_dtype)
  chunks, gpu_ms = run_batch_stage(model, [obs_cpu], device, stream, autocast_dtype,
                                   offload=offload)
  return chunks[0], gpu_ms


def reply_time(gpu_done, offload):
  """When a batch that left the GPU at gpu_done may be answered: the pass end
  itself, or that plus the simulated PNM latency when offloaded."""
  return gpu_done if offload is None else offload.reply_time(gpu_done)


def build_warmup_obs_cpu(model_choice, benchmark):
  """A CPU observation matching exactly what a client sends, for warmup."""
  preprocess, spec = build_preprocessor(model_choice, benchmark, torch.device("cpu"))
  return build_observation_batch(preprocess, spec, torch.device("cpu"))


def warmup_shapes(model, warmup_obs_cpu, device, autocast_dtype, crit_batch_max, batch_max,
                  passes=5, offload=None, timed_passes=5):
  """Compile every shape the priority server will serve, one at a time on the main
  thread, before any worker starts. Two workers compiling concurrently would race
  the compiler; this also keeps the first real request off the compile path.

  Returns {"critical": s, "normal": s}: the measured cost of ONE pass at each
  shape, in seconds. The scheduler is built on 'how long does a batch occupy the
  GPU', and that is a property of (model, shape, device) spanning roughly 4x
  between the models here -- a constant tuned on one of them silently mis-sizes
  the other. Measuring it during a warmup that already runs the exact shapes
  costs a few extra passes and removes the guess.

  The timed passes run AFTER the warmup passes, so compile and per-stream
  first-touch are already paid, and they are timed exactly the way the workers
  time themselves (wall time around the same call, whose internal
  stream.synchronize() is what makes it a real pass cost)."""
  stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

  def one_pass(batch):
    if batch > 1:
      run_batch_stage(model, [warmup_obs_cpu] * batch, device, stream,
                      autocast_dtype, pad_to=batch, offload=offload)
    else:
      run_stage(model, warmup_obs_cpu, device, stream, autocast_dtype, offload=offload)

  measured = {}
  for label, batch in (("critical", crit_batch_max), ("normal", batch_max)):
    t0 = time.perf_counter()
    for _ in range(passes):
      one_pass(batch)
    if device.type == "cuda":
      torch.cuda.synchronize()
    warm_s = time.perf_counter() - t0

    samples = []
    for _ in range(timed_passes):
      t1 = time.perf_counter()
      one_pass(batch)
      samples.append(time.perf_counter() - t1)
    # MIN, not median: allocator growth, kernel loading and any external GPU load
    # only ever ADD time, so the fastest sample is the best estimate of the real
    # pass cost. A median over few samples tracked the warm-up curve instead and
    # read 3-4x high, which then seeded the gate's cost model wrong.
    measured[label] = min(samples)

    print(f"Warmup {label} shape (batch={batch}): {warm_s:.1f}s warm, "
          f"pass = {1000 * measured[label]:.1f} ms "
          f"(min of {timed_passes}, max {1000 * max(samples):.1f})")
  return measured


def warmup_sizes(model, warmup_obs_cpu, device, autocast_dtype, batch_max, passes=3,
                 timed_passes=5, offload=None):
  """Compile and time every UNPADDED batch size 1..batch_max, for unpadded Armory.

  Returns {b: seconds}, the min of timed_passes at each size (see warmup_shapes
  for why min). Run after force_static_compile, so each size is its own static
  graph rather than one dynamic one. With an offload, the GPU's prefill only."""
  stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
  measured = {}
  for b in range(1, batch_max + 1):
    obs = [warmup_obs_cpu] * b
    t0 = time.perf_counter()
    for _ in range(passes):
      run_batch_stage(model, obs, device, stream, autocast_dtype, offload=offload)
    samples = []
    for _ in range(timed_passes):
      t1 = time.perf_counter()
      run_batch_stage(model, obs, device, stream, autocast_dtype, offload=offload)
      samples.append(time.perf_counter() - t1)
    measured[b] = min(samples)
    print(f"Warmup batch={b}: {time.perf_counter() - t0:.1f}s, "
          f"pass = {1000 * measured[b]:.1f} ms (min of {timed_passes})")
  return measured


def force_static_compile(model, compile_mode):
  """Re-wrap the policy's compiled sample_actions with dynamic=False.

  A server whose critical and normal batch shapes differ runs TWO shapes in one
  process. lerobot compiles sample_actions without pinning dynamic, so the second
  shape makes dynamo recompile a DYNAMIC graph -- and on sm70 that graph's Triton
  kernel dies with 'CUDA driver error: an illegal memory access' (reproduced on
  pi0). dynamic=False forces a separate static graph per shape."""
  inner = getattr(model, "model", None)
  fn = getattr(inner, "sample_actions", None)
  if fn is None:
    return False
  # Unwrap an already-compiled callable so we recompile the original function.
  original = getattr(fn, "_torchdynamo_orig_callable", fn)
  inner.sample_actions = torch.compile(original, mode=compile_mode, dynamic=False)
  return True


def force_static_prefill(model, compile_mode):
  """The same for the policy's vlm_prefill, which the unpadded PNM arms run at
  every batch size: one static graph per size instead of a dynamic one."""
  fn = getattr(model, "vlm_prefill", None)
  if fn is None:
    return False
  original = getattr(fn, "_torchdynamo_orig_callable", fn)
  model.vlm_prefill = torch.compile(original, mode=compile_mode, dynamic=False)
  return True
