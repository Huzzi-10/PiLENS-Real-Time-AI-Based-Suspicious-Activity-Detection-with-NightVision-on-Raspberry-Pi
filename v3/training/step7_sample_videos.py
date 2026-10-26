"""PiLENS Step 7 (Kaggle): picks short real UCF-Crime test videos to try the Pi pipeline on.
Chooses 6 anomaly videos (different classes, the shortest) and 4 normal videos (the shortest) from Anomaly_Test.txt,
copies them to /kaggle/working/samples/ and writes /kaggle/working/download/samples.zip.
Paste into one cell and run (no GPU needed, about 1 minute).
"""
import os, shutil, zipfile, collections

ROOT = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset"
OUT, ZIP = "/kaggle/working/samples", "/kaggle/working/download/samples.zip"
N_ANOM, N_NORM, MAX_MB = 6, 4, 20

sp = os.path.join(ROOT, "UCF_Crimes-Train-Test-Split", "Anomaly_Detection_splits", "Anomaly_Test.txt")
want = {}
for l in open(sp, errors="ignore"):
    t = l.split()
    if t:
        want[os.path.basename(t[0])] = t[0].split("/")[0]            # video file -> class folder (Normal_Videos_event, Fighting, ...)

found = {}
for dp, _, fs in os.walk(ROOT):
    for f in fs:
        if f in want and f not in found:
            found[f] = os.path.join(dp, f)
print("test list:", len(want), "| disk par mile:", len(found))

rows = sorted((os.path.getsize(p), f, want[f]) for f, p in found.items() if os.path.getsize(p) <= MAX_MB * 1e6)
anom, seen = [], set()
for sz, f, c in rows:
    if "Normal" not in c and c not in seen and len(anom) < N_ANOM:
        anom.append((sz, f, c)); seen.add(c)
norm = [r for r in rows if "Normal" in r[2]][:N_NORM]

shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT); os.makedirs(os.path.dirname(ZIP), exist_ok=True)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:
    for sz, f, c in anom + norm:
        tag = "anom" if "Normal" not in c else "normal"
        name = f"{tag}_{c}_{f}"
        shutil.copy(found[f], os.path.join(OUT, name)); z.write(os.path.join(OUT, name), name)
        print(f"{name:55s} {sz / 1e6:5.1f} MB")
print(f"\n{ZIP}  {os.path.getsize(ZIP) / 1e6:.1f} MB")
print("Is zip ko Kaggle Output se download karo (ya samples/ folder ki files alag alag), phir yahan bhejo ya Pi par rakho.")
print("END OF REPORT")
