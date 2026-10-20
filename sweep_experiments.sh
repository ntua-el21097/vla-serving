#!/usr/bin/env bash
#
# sweep_experiments.sh -- experiment sweep for the VLA policy server.
#
# Drives orchestrator.py across (model:benchmark) pairs x serving arms x fleet
# sizes x critical fractions x arrival rates x seeds, always compiled at
# $PRECISION. Every run lands under one timestamped sweep directory with its
# CSVs, and per-run metrics are collected into summary.csv.
#
# Each model is pinned to its own benchmark (MODEL_BENCH), because the camera
# setups are not interchangeable between models in this study.
#
# Ordering note: torch.compile is keyed on (model, input shape) -> only the FIRST
# run of each pair pays the compile (~4 min smolvla / ~9 min pi0); every later
# fleet/rate combo reuses the Inductor disk cache. num_clients and arrival rate do
# NOT change the compiled shape, so they are free to vary against a cached
# compile; BATCH_MAX and CRIT_BATCH_MAX do, and each new value pays once.
#
# Run from the project root with the lerobot env active. Usage:
#   ./sweep_experiments.sh                                    # the pi0 evaluation
#   SEEDS="1 2" ./sweep_experiments.sh                        # override any knob
#
set -uo pipefail

# ------------------------- config (edit or override via env) -------------------------
# Defaults are the thesis evaluation (pi0, bf16); for SmolVLA set
# MODEL_BENCH and PRECISION=fp16.
# model:benchmark pairs -- each model runs ONLY against its own camera setup.
MODEL_BENCH="${MODEL_BENCH:-pi0:2_cameras}"
# pi0's config takes float32/bfloat16 and never float16, so on sm80+ it wants
# bf16; smolvla has no dtype field and loads at a pinned fp16. One sweep
# invocation per precision when the two models are run together.
PRECISION="${PRECISION:-bf16}"                    # fp16 | bf16 | fp32
# Explicit CPU split, passed straight through to the orchestrator. Left empty it
# derives one from the NUMA topology, which on fury0 hands the server the whole
# 12-core node its GPU sits on -- cores the server cannot use -- while the
# clients are squeezed onto the other 12. Setting it frees cores for clients.
SERVER_CPUS="${SERVER_CPUS:-}"                    # e.g. "12,13,14,15"
CLIENT_CPUS="${CLIENT_CPUS:-}"                    # e.g. "0,1,...,11,16,...,23"
FLEET_SIZES="${FLEET_SIZES:-10 20}"               # concurrent robots (clients) = fleet size / load
CRITICAL_FRACTIONS="${CRITICAL_FRACTIONS:-0.1 0.2 0.3}"  # share of the fleet on critical duty
ARRIVAL_RATES="${ARRIVAL_RATES:-10}"              # Poisson robots/s for the opening wave
SEEDS="${SEEDS:-0}"                               # one seed per repetition
# Serving policy. Baselines: 'batching' (dynamic batcher) and 'prio-streams'
# (priority routing only -- no prediction, gate, exclusion or work stealing).
# Ours: 'aegis', and 'aegis-pnm' (that scheduler with the action expert offloaded).
# Ported from Armory: 'armory' (Lookahead), 'armory-pnm' (with the offload) and
# 'edf'. Every arm pads its passes to BATCH_MAX.
SERVINGS="${SERVINGS:-batching prio-streams edf armory armory-pnm aegis aegis-pnm}"
ARMORY_CRIT_WEIGHT="${ARMORY_CRIT_WEIGHT:-5}"  # armory/armory-pnm: Lookahead weight of critical robots (5 = LA@5)
ARMORY_UNPADDED="${ARMORY_UNPADDED:-0}"         # every arm: 1 = each batch at its own size
PNM_DENOISE_MS="${PNM_DENOISE_MS:-}"           # aegis-pnm/armory-pnm: override the per-model simulated PNM latency
BATCH_MAX="${BATCH_MAX:-5}"                        # max requests per normal batch
BATCH_TIMEOUT_MS="${BATCH_TIMEOUT_MS:-200}"       # deadline (ms) to fire a partial normal batch
CHUNK_LEN="${CHUNK_LEN:-50}"                      # actions every robot plays from each chunk
CONTROL_HZ="${CONTROL_HZ:-30}"                    # best-effort robots: action playback rate
H_MAX="${H_MAX:-50}"                              # best-effort: most actions played per chunk
EXEC_HORIZON="${EXEC_HORIZON:-35}"                # best-effort: request point (from the chunk's first action)
CRIT_CONTROL_HZ="${CRIT_CONTROL_HZ:-60}"          # critical robots: action playback rate
CRIT_H_MAX="${CRIT_H_MAX:-40}"                    # critical: most actions played per chunk
CRIT_EXEC_HORIZON="${CRIT_EXEC_HORIZON:-25}"      # critical: request point (H_max - 25 = 15 actions of buffer)
INFERENCE_STEPS="${INFERENCE_STEPS:-7}"           # observations each best-effort robot sends (<=20)
CRIT_INFERENCE_STEPS="${CRIT_INFERENCE_STEPS:-20}"  # observations each critical robot sends (<=20)
# aegis fills its padding slots by work stealing, so a critical batch of 5 is free
# and keeps one compiled shape. prio-streams cannot steal: at 5 its critical stream
# never filled, so it keeps 2.
CRIT_BATCH_MAX="${CRIT_BATCH_MAX:-5}"          # aegis/aegis-pnm: critical batch shape
PRIOBASE_CRIT_BATCH_MAX="${PRIOBASE_CRIT_BATCH_MAX:-2}"  # prio-streams: critical batch shape
GROUP_CRITICALS="${GROUP_CRITICALS:-1}"        # 1 = critical robots start in groups of BATCH_MAX, 0 = at their own random starts
CRIT_FIRST_CORES="${CRIT_FIRST_CORES:-0}"      # 1 = critical robots take the first CLIENT_CPUS entries
CRIT_GATHER_MS="${CRIT_GATHER_MS:-35}"         # aegis: window a robot's first request waits for its group
# prio-streams uses a blind linger, not a predictive wait, so it needs a longer
# window to batch criticals at all; given separately so each is tuned on its own.
PRIOBASE_GATHER_MS="${PRIOBASE_GATHER_MS:-50}" # prio-streams: its critical stream's deadline
CRIT_STREAM_PRIO="${CRIT_STREAM_PRIO:--1}"     # CUDA stream priority of the critical worker (-1 = high band)
GATE_JITTER_SIGMAS="${GATE_JITTER_SIGMAS:-2}"  # aegis/aegis-pnm: gate keeps this many sigmas of arrival jitter free
RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-5400}"         # hard cap per run, incl. model load + compile
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-3600}"  # model load + warmup/compile allowance
OUT_ROOT="${OUT_ROOT:-pi0_eval_results}"
# ------------------------------------------------------------------------------------

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="$OUT_ROOT/sweep_${STAMP}"
LOG_DIR="$SWEEP_DIR/logs"
mkdir -p "$LOG_DIR"
SUMMARY="$SWEEP_DIR/summary.csv"
# *_crit / *_norm split the same requests by the observation's priority flag.
# wait_* is robot_wait_ms (the robot out of actions and idle); step 0 is excluded
# because a robot has no chunk yet, so its first wait is the whole round trip by
# construction. starve_* converts wait to Armory's unit: idle control ticks over
# all control ticks.
echo "model,benchmark,fleet,critical_fraction,control_hz,crit_control_hz,exec_horizon,crit_exec_horizon,batch_max,arrival_rate,requests,actions_per_s,gpu_avg_ms,total_p50_ms,total_p90_ms,total_p99_ms,rtt_p50_ms,rtt_p90_ms,rtt_p99_ms,rtt_crit_p50_ms,rtt_crit_p90_ms,rtt_norm_p50_ms,rtt_norm_p90_ms,n_crit,seed,wait_crit_mean_ms,wait_crit_p50_ms,wait_crit_p90_ms,wait_crit_p99_ms,wait_crit_frac_gt0,wait_norm_mean_ms,wait_norm_p90_ms,starve_crit,starve_norm,starve_all,wall_s,serving,batch_avg,reqs_per_gpu_s,run_dir" > "$SUMMARY"

# Count total runs for progress display.
TOTAL=0
for _p in $MODEL_BENCH; do for _sv in $SERVINGS; do for _f in $FLEET_SIZES; do for _c in $CRITICAL_FRACTIONS; do for _a in $ARRIVAL_RATES; do for _sd in $SEEDS; do TOTAL=$((TOTAL+1)); done; done; done; done; done; done
echo "=== experiment sweep: $TOTAL runs ($PRECISION + compiled) ==="
echo "    pairs=[$MODEL_BENCH]  fleets=[$FLEET_SIZES]  critical_fractions=[$CRITICAL_FRACTIONS]  arrival_rates=[$ARRIVAL_RATES]"
echo "    serving=$SERVINGS (batch_max=$BATCH_MAX, crit_batch_max=$CRIT_BATCH_MAX, prio-streams crit_batch_max=$PRIOBASE_CRIT_BATCH_MAX)  seeds=[$SEEDS]"
echo "    best-effort ${CONTROL_HZ}Hz/h${EXEC_HORIZON}  critical ${CRIT_CONTROL_HZ}Hz/h${CRIT_EXEC_HORIZON}"
echo "    output: $SWEEP_DIR"
echo

PRIO_FLAGS="--gate-jitter-sigmas $GATE_JITTER_SIGMAS"
[ -n "$PNM_DENOISE_MS" ] && PRIO_FLAGS="$PRIO_FLAGS --pnm-denoise-ms $PNM_DENOISE_MS"
[ "$GROUP_CRITICALS" = "0" ] && PRIO_FLAGS="$PRIO_FLAGS --no-group-criticals"
[ "$CRIT_FIRST_CORES" = "1" ] && PRIO_FLAGS="$PRIO_FLAGS --crit-first-cores"
[ "$ARMORY_UNPADDED" = "1" ] && PRIO_FLAGS="$PRIO_FLAGS --armory-unpadded"

SWEEP_START=$(date +%s)
i=0
# Pair is the outer dimension so each model compiles once, then every fleet x rate
# combo under it is an Inductor disk-cache hit.
for pair in $MODEL_BENCH; do
  model="${pair%%:*}"
  bench="${pair##*:}"
  # Arms innermost: every (fleet, fraction, rate, seed) cell runs all arms back
  # to back, so they are compared under the same conditions and drift.
  for fleet in $FLEET_SIZES; do
   for cfrac in $CRITICAL_FRACTIONS; do
    for rate in $ARRIVAL_RATES; do
      for SEED in $SEEDS; do
  for SERVING in $SERVINGS; do
        i=$((i+1))
        gather_ms="$CRIT_GATHER_MS"
        crit_bmax="$CRIT_BATCH_MAX"
        case "$SERVING" in
          batching)     serve_tag="b${BATCH_MAX}x1" ;;
          prio-streams) gather_ms="$PRIOBASE_GATHER_MS"
                        crit_bmax="$PRIOBASE_CRIT_BATCH_MAX"
                        serve_tag="priobase${crit_bmax}g${PRIOBASE_GATHER_MS}" ;;
          aegis)        serve_tag="aegis${BATCH_MAX}" ;;
          aegis-pnm)    serve_tag="aegispnm${BATCH_MAX}" ;;
          armory)       serve_tag="armory${ARMORY_CRIT_WEIGHT}_${BATCH_MAX}" ;;
          armory-pnm)   serve_tag="armorypnm${ARMORY_CRIT_WEIGHT}_${BATCH_MAX}" ;;
          edf)          serve_tag="edf${BATCH_MAX}" ;;
        esac
        case "$SERVING" in armory|edf|batching|prio-streams|aegis|aegis-pnm|armory-pnm) [ "$ARMORY_UNPADDED" = "1" ] && serve_tag="${serve_tag}u" ;; esac
        tag="${model}_${bench}_${fleet}c_cf${cfrac}_ar${rate}_${CONTROL_HZ}hz_c${CRIT_CONTROL_HZ}hz_${serve_tag}_s${SEED}"
        log="$LOG_DIR/${tag}.log"
        printf "[%2d/%2d] %-7s %-9s fleet=%-3s cf=%-4s rate=%-3s seed=%s %-7s ... " \
          "$i" "$TOTAL" "$model" "$serve_tag" "$fleet" "$cfrac" "$rate" "$SEED" "$SERVING"
        run_start=$(date +%s)

        # Outermost backstop. The orchestrator and server have their own timeouts;
        # this catches anything below them (a wedged GPU leaves processes in
        # uninterruptible sleep, where no Python timeout can run at all).
        timeout --kill-after=60 "$RUN_TIMEOUT_S" \
        python -u orchestrator.py \
          --model "$model" --benchmark "$bench" \
          --precision "$PRECISION" --compile \
          --serving "$SERVING" \
          --batch-max "$BATCH_MAX" --batch-timeout-ms "$BATCH_TIMEOUT_MS" \
          --num-clients "$fleet" \
          --control-hz "$CONTROL_HZ" --exec-horizon "$EXEC_HORIZON" \
          --crit-control-hz "$CRIT_CONTROL_HZ" --crit-exec-horizon "$CRIT_EXEC_HORIZON" \
          --chunk-len "$CHUNK_LEN" --h-max "$H_MAX" --crit-h-max "$CRIT_H_MAX" \
          --inference-steps "$INFERENCE_STEPS" --crit-inference-steps "$CRIT_INFERENCE_STEPS" \
          --arrival-rate "$rate" --seed "$SEED" \
          --critical-fraction "$cfrac" --crit-batch-max "$crit_bmax" \
          --crit-gather-ms "$gather_ms" \
          --crit-stream-priority "$CRIT_STREAM_PRIO" ${PRIO_FLAGS} \
          --armory-crit-weight "$ARMORY_CRIT_WEIGHT" \
          --server-ready-timeout "$SERVER_READY_TIMEOUT_S" \
          ${SERVER_CPUS:+--server-cpus "$SERVER_CPUS"} \
          ${CLIENT_CPUS:+--client-cpus "$CLIENT_CPUS"} \
          --out-dir "$SWEEP_DIR" \
          > "$log" 2>&1
        rc=$?
        run_wall=$(( $(date +%s) - run_start ))

        if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
          echo "TIMED OUT after ${RUN_TIMEOUT_S}s (see $log)"
          continue
        fi
        if [ $rc -ne 0 ]; then
          echo "FAILED (rc=$rc, see $log)"
          continue
        fi

        # Locate the run dir the orchestrator created (printed in its summary).
        run_dir="$(grep -m1 'artifacts' "$log" | sed 's/.*: //')"
        if [ -z "$run_dir" ] || [ ! -f "$run_dir/server.csv" ]; then
          echo "done but no server.csv (see $log)"
          continue
        fi

        # Extract per-run metrics straight from the CSVs (robust vs parsing text).
        row="$(python - "$run_dir" "$run_wall" <<'PY'
import csv, glob, json, os, sys
import numpy as np

rd, wall = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(rd, "config.json")))

def col(path, name):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.append(float(r[name]))
    return np.array(out)

s = os.path.join(rd, "server.csv")
tot, gpu = col(s, "server_total_ms"), col(s, "gpu_time_ms")

def reqs_per_gpu_s(path):
    """Requests served per second of GPU time: each pass counted once (its rows
    share a completion time and queue), by its GPU wall time (prefill only under
    a PNM offload). Server-side, so independent of how often robots ask."""
    passes, n = {}, 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n += 1
            passes.setdefault((row["completion_time_s"], row["queue"] == "normal"),
                              float(row["inference_wall_ms"]) / 1000.0)
    busy = sum(passes.values())
    return round(n / busy, 2) if busy > 0 else "nan"

def batch_avg(path):  # mean batch size over requests
    with open(path, newline="") as f:
        vals = [float(row["batch_size"]) for row in csv.DictReader(f)]
    return sum(vals) / len(vals) if vals else 1.0

cols = ("infer_rtt_ms", "priority", "robot_wait_ms", "step", "n_actions", "control_hz",
        "send_time_s", "recv_time_s", "exec_ms")
C = {k: [] for k in cols}
# Throughput per robot, summed over the fleet: each robot's executed actions over
# its own active time. A whole-run span would mostly measure the opening wave and
# the last robot to finish, since every robot sends a fixed number of requests.
# No request rate: a robot whose request queues asks later, so requests/s falls
# while each chunk delivers more -- it measures how often robots ask, not service.
act_rate = 0.0
for c in sorted(glob.glob(os.path.join(rd, "client*.csv"))):
    one = {k: col(c, k) for k in cols}
    for k in cols:
        C[k] += list(one[k])
    if one["step"].size:
        active = one["recv_time_s"][-1] + one["exec_ms"][-1] / 1000 - one["send_time_s"][0]
        if active > 0:
            act_rate += one["n_actions"].sum() / active
C = {k: np.array(v) for k, v in C.items()}
nan = np.array([float("nan")])
is_crit, is_norm = C["priority"] == 1, C["priority"] == 0
later = C["step"] > 0

def sel(a, m):
    return a[m] if m.any() else nan

def p(a, q):
    a = a[~np.isnan(a)] if a.size else a
    return round(float(np.percentile(a, q)), 2) if a.size else "nan"

def mean(a):
    a = a[~np.isnan(a)]
    return round(float(a.mean()), 2) if a.size else "nan"

def starve(m):
    """Idle control ticks / all control ticks, over the cycles selected by m."""
    m = m & later
    idle = C["robot_wait_ms"][m] / 1000.0 * C["control_hz"][m]
    total = C["n_actions"][m] + idle
    return round(float(idle.sum() / total.sum()), 4) if total.sum() > 0 else "nan"

rtts = C["infer_rtt_ms"] if C["infer_rtt_ms"].size else nan
actions_per_s = round(act_rate, 2)
wc, wn = sel(C["robot_wait_ms"], is_crit & later), sel(C["robot_wait_ms"], is_norm & later)
# > 1 ms, not > 0: collecting a reply that is already there records ~0.01 ms.
gt0 = round(float((wc > 1.0).mean()), 4) if not np.isnan(wc).all() else "nan"

print(",".join(str(x) for x in [
    cfg["model"], cfg["benchmark"], cfg["num_clients"], cfg["critical_fraction"],
    cfg["control_hz"], cfg["crit_control_hz"], cfg["exec_horizon"], cfg["crit_exec_horizon"],
    cfg.get("batch_max", "na"), cfg.get("arrival_rate", "na"),
    len(tot), actions_per_s, round(float(gpu.mean()), 2),
    p(tot, 50), p(tot, 90), p(tot, 99),
    p(rtts, 50), p(rtts, 90), p(rtts, 99),
    p(sel(rtts, is_crit), 50), p(sel(rtts, is_crit), 90),
    p(sel(rtts, is_norm), 50), p(sel(rtts, is_norm), 90),
    int(is_crit.sum()), cfg.get("seed", 0),
    mean(wc), p(wc, 50), p(wc, 90), p(wc, 99), gt0, mean(wn), p(wn, 90),
    starve(is_crit), starve(is_norm), starve(is_crit | is_norm),
    wall, cfg.get("serving", "batching"), round(batch_avg(s), 2), reqs_per_gpu_s(s), rd,
]))
PY
)"
        if [ -n "$row" ]; then
          echo "$row" >> "$SUMMARY"
          aps="$(echo "$row" | cut -d, -f12)"
          wc90="$(echo "$row" | cut -d, -f28)"
          sc="$(echo "$row" | cut -d, -f33)"
          sn="$(echo "$row" | cut -d, -f34)"
          echo "ok  ${run_wall}s  |  ${aps} actions/s  |  starved: crit ${sc}, best-effort ${sn}  |  crit wait p90=${wc90}ms"
        else
          echo "done but metric extraction failed (see $log)"
        fi
      done
    done
   done
  done
  done
done

SWEEP_WALL=$(( $(date +%s) - SWEEP_START ))
echo
echo "=== sweep complete in ${SWEEP_WALL}s ==="
echo "summary: $SUMMARY"
echo
column -t -s, "$SUMMARY"
