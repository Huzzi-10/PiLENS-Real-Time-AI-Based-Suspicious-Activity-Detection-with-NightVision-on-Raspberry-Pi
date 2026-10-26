"""PiLENS Step 8 (Kaggle): packs the results the paper needs into a small zip (X3D and MoViNet outputs found in /kaggle/working).

Run in one cell. Creates:
  /kaggle/working/download/paper_pack.zip : binary_out*/{eval_pack.npz,test_meta.json,*.json,norm.npz}, cls14_out*/{cls14_pack.npz,
                                            cls14_meta.json,*.json,confusion*.csv,norm_det.npz}, video_meta.csv  (a few MB)
  /kaggle/working/download/export_<tag>.zip : every export*/ folder (ONNX + meta, for the Pi)
Prints the zip sizes at the end. Download paper_pack.zip from the Kaggle Output tab.
"""
import os, zipfile, glob, shutil

W = "/kaggle/working"; D = os.path.join(W, "download")
KEEP = ("eval_pack.npz", "test_meta.json", "cls14_pack.npz", "cls14_meta.json", "norm.npz", "norm_det.npz", "ablation_results.json",
        "binary_results.json", "cls14_results.json", "heads_meta.json")


def main():
    shutil.rmtree(D, ignore_errors=True); os.makedirs(D)
    out = os.path.join(D, "paper_pack.zip")
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d in sorted(glob.glob(os.path.join(W, "binary_out*")) + glob.glob(os.path.join(W, "cls14_out*"))):
            for f in sorted(os.listdir(d)):
                if f in KEEP or f.startswith("confusion"):
                    z.write(os.path.join(d, f), os.path.join(os.path.basename(d), f)); n += 1
        if os.path.exists(os.path.join(W, "video_meta.csv")):
            z.write(os.path.join(W, "video_meta.csv"), "video_meta.csv"); n += 1
    print(f"paper_pack.zip: {os.path.getsize(out) / 1e6:.2f} MB | {n} files | integrity {'OK' if zipfile.ZipFile(out).testzip() is None else 'FAIL'}")
    for d in sorted(glob.glob(os.path.join(W, "export*"))):
        if os.path.isdir(d):
            p = shutil.make_archive(os.path.join(D, "export_" + os.path.basename(d).replace("export", "").strip("_") or "x3ds"), "zip", d)
            print(f"{os.path.basename(p)}: {os.path.getsize(p) / 1e6:.1f} MB")
    print("\nEND OF REPORT")


main()
