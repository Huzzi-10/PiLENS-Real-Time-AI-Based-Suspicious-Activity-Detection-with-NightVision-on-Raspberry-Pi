#!/bin/bash
# 4 models (A0, A0 gray, A1, X3D) ko ek ke baad ek, wahi sample clips par chalata hai aur har ek ki screen video (demo.mp4) + report banata hai.
# Chalane ka tareeqa (PiLENS folder mein):   bash demo_all.sh
# Kul ~12-15 min, khud chalta hai. Window khulti hai: use band mat karo, mouse mat chhero, Ctrl+C mat dabao.
cd "$(dirname "$0")"; source venv/bin/activate
for D in export_movinet export_movinet_a0_gray export_movinet_a1 export; do
  if [ -d "$D" ]; then echo "=== $D ==="; python pilens_demo.py --source samples --dir "$D" --record; sleep 5
  else echo "SKIP: $D folder nahi mila"; fi
done
echo "KHATAM. Ab demo_out/ mein 4 folder hain (har ek mein demo.mp4, report.txt, scores.csv, summary.png)."
