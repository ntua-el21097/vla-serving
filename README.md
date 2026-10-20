# VLA serving with critical robots

An inference server for vision-language-action policies (π0, SmolVLA) serving a fleet of
robots, some of which are critical. It implements our scheduler and the baselines it is
compared against, plus a closed-loop robot simulator that drives them.

## Serving arms

| `--serving` | Arm | What it does |
|---|---|---|
| `batching` | FIFO | One dynamic batcher on one CUDA stream; no notion of priority. |
| `prio-streams` | CUDA-Priority | Critical requests get their own queue and worker on a high-priority CUDA stream. |
| `edf` | EDF | Armory's `GreedyDeadlineScheduler`, ported. |
| `armory` | Armory | Armory's `Lookahead`, ported (critical weight 5). |
| `aegis` | Ours | Arrival prediction, a predictive admission gate, GPU exclusion and work stealing. |
| `aegis-pnm` | Ours-PIM | Ours, with the action expert offloaded to near-memory processing (simulated latency). |
| `armory-pnm` | Armory-PIM | Armory with the same offload. |

EDF and Armory port only the scheduling decision from
[GaTech-RL2/armory](https://github.com/GaTech-RL2/armory) at `e876202`; see the notes in
`vla_serving/scheduling.py`. Every arm serves a per-task action expert per request
(`vla_serving/hetero_expert.py`).

## Layout

- `policy_server.py`: the gRPC inference server; `--serving` picks the arm.
- `vla_serving/`: scheduling policies (`scheduling.py`), forward pass and compilation
  (`execution.py`), per-task experts (`hetero_expert.py`), gRPC handler and logging
  (`serving.py`).
- `robot_client.py`: one simulated robot (closed loop, action-chunk playback).
- `orchestrator.py`: one experiment: a server plus a fleet of robots.
- `sweep_experiments.sh`: a grid of experiments, collected into `summary.csv`.
- `grpc_api/`: the robot–server protocol (`inference.proto` and generated stubs).
- `lerobot/`: the lerobot 0.6.1 source this code runs against, modified to add
  `vlm_prefill` to the π0 policy (used by the PIM arms and the scheduler's cost model).
  lerobot is Apache-2.0 licensed; see `lerobot/LICENSE`.

## Requirements

- Python 3.12, PyTorch 2.10 (CUDA 12.8), `grpcio`, `protobuf`, `numpy`.
- An NVIDIA GPU (the evaluation used an RTX A6000).
- The bundled lerobot, installed in editable mode:

  ```bash
  pip install -e ./lerobot
  ```

## Running

Run from the repository root with the environment active.

```bash
export VLA_LIBRARY_CUMSUM=1     # compile cumsum as a library call
./sweep_experiments.sh          # every setting can be overridden through the environment
```

The settings are listed at the top of `sweep_experiments.sh`. The final RTX A6000
evaluation used:

```bash
MODEL_BENCH=pi0:2_cameras PRECISION=bf16 \
SERVINGS="batching prio-streams edf armory aegis" ARMORY_UNPADDED=1 \
BATCH_MAX=3 CRIT_BATCH_MAX=2 PRIOBASE_CRIT_BATCH_MAX=1 BATCH_TIMEOUT_MS=5 \
FLEET_SIZES=10 CRITICAL_FRACTIONS="0.1 0.2 0.3" SEEDS="0 1 2 3" \
CONTROL_HZ=30 EXEC_HORIZON=5 H_MAX=20 CRIT_CONTROL_HZ=30 CRIT_EXEC_HORIZON=5 CRIT_H_MAX=10 \
INFERENCE_STEPS=20 CRIT_INFERENCE_STEPS=64 GROUP_CRITICALS=0 GATE_JITTER_SIGMAS=1 \
CRIT_FIRST_CORES=1 SERVER_CPUS=0,1,2,3 CLIENT_CPUS=<one thread per remaining core> \
./sweep_experiments.sh
```

The PIM arms ran the same way with `SERVINGS="armory-pnm aegis-pnm" BATCH_MAX=2`.

## Output

Each run writes a folder with:

- `server.csv`: one row per request, measured at the server.
- `clientNN.csv`: one row per request, measured at each robot.
- `config.json`: the run's settings.
- `prediction_errors.csv` and `gate_stats.json`: our arms only.
