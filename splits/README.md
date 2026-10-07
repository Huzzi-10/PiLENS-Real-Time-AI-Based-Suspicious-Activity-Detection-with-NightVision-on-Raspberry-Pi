# splits/

Generated on Kaggle by `python -m pilens_v2.data.splits --root <dataset> --out splits --probe`
(see `docs/v2_quickstart.md`, Step 2) and committed so every experiment uses the same
video-level lists:

- `video_index.csv`: one row per unique video (duplicates removed), class, path, frame count, fps
- `duplicates.csv`: byte-identical copies that were dropped
- `cls14_fold{1..4}_{train,val,test}.txt`: official 14-class folds, 15% stratified val from train
- `anomaly_{train,val,test}.txt`: official 1610 / 290 anomaly split, 15% stratified val from train
- `day_night.csv`: manual + automatic Day/Night labels (`pilens_v2.data.daynight`)
- `split_summary.json`: counts, which list files were used / skipped, leak check
