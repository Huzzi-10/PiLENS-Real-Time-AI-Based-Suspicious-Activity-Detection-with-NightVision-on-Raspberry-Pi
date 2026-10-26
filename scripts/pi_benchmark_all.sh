#!/usr/bin/env bash
# PiLENS v2: every Pi measurement for the paper (roadmap Steps 5, 6 and 9), in order.
#   source ~/pilens-venv/bin/activate && bash scripts/pi_benchmark_all.sh
# Takes ~2.5 hours (two 30-min thermal runs + two 30-min live runs). Results:
#   bench_results/*.json, logs/*/summary.json, docs/v2_results.md  -> commit these.
set -euo pipefail
cd "$(dirname "$0")/.."
EX=${EXPORTS:-exports}
LONG=${LONG_SEC:-1800}

wait_cool() {  # start every run from the same temperature
  local target=${1:-50} temp
  while true; do
    temp=$(( $(cat /sys/class/thermal/thermal_zone0/temp) / 1000 ))
    [ "$temp" -le "$target" ] && break
    echo "   cooling down: ${temp} C (waiting for <= ${target} C)"; sleep 30
  done
}
pause() { read -r -p ">> $1  [Enter dabao] " _; }

pause "FAN / heatsink LAGA hua hai? (fan wale runs)"
echo "== 1. FP32 clip latency, threads 1-4"
for t in 1 2 3 4; do
  wait_cool; python -m pilens_v2.bench.benchmark --exports "$EX" --threads "$t" --tag "fp32_fan_t$t"
done
echo "== 2. FP32 30-min thermal run (fan)"
wait_cool; python -m pilens_v2.bench.benchmark --exports "$EX" --duration "$LONG" --tag fp32_fan_30min

echo "== 3. Live pipeline (camera + motion gate + alert thread), fan"
wait_cool; python -m pilens_v2.runtime.run --exports "$EX" --no-email --duration "$LONG" --log-dir logs/live_fan_gate
wait_cool; python -m pilens_v2.runtime.run --exports "$EX" --no-email --no-motion --duration 600 --log-dir logs/live_fan_nogate
wait_cool; python -m pilens_v2.runtime.run --exports "$EX" --no-email --no-stage2 --duration 600 --log-dir logs/live_fan_binary_only

if [ -f "$EX/backbone_int8.onnx" ]; then
  echo "== 4. INT8 clip latency"
  wait_cool; python -m pilens_v2.bench.benchmark --exports "$EX" --backbone backbone_int8.onnx --threads 4 --tag int8_fan_t4
fi

pause "Ab FAN HATAO (no-cooling runs)"
echo "== 5. FP32 30-min thermal run, no fan"
wait_cool; python -m pilens_v2.bench.benchmark --exports "$EX" --duration "$LONG" --tag fp32_nofan_30min

python -m pilens_v2.eval.make_tables --out docs/v2_results.md > /dev/null
echo "== done: docs/v2_results.md, bench_results/, logs/ commit karo"
