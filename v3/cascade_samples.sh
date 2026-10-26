#!/bin/bash
# samples/ ki har video par A0 + CLIP cascade (asli speed par, camera ki tarah). Chalao: bash cascade_samples.sh
cd "$(dirname "$0")"; source venv/bin/activate
echo "SCRIPT VERSION: cascade_samples v1.0"
for V in samples/*; do
  N=$(basename "$V"); echo "=== $N"
  python step6_cascade.py --dir export_movinet --source "$V" --quiet --out "out_cascade/$N" 2>&1 \
    | grep -E "== SUMMARY|A0 ALERT|CLIP p=|^youden|^fpr10|^fpr5|PROBLEM|NAHI|Error|error" 
done
echo "END OF REPORT"
