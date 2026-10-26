# Other backbones (same protocol as MoViNet-A0)

Official UCF-Crime split, thresholds from validation only, test set used once, 5 seeds (mean ± std). Each set of backbones was run with the same evaluation script.

| experiment | pretrained | params | video AUC | recall / false-alarm | 13 classes acc (chance 7.7) | top-3 | 4 groups acc (chance 31.4) |
|---|---|---|---|---|---|---|---|
| mnv3_tsm_scratch | scratch | 3.0M | 0.804+-0.002 | 64.9+-0.7% / 21.1+-0.7% | 12.9+-3.7% | 34.0+-5.0% | 31.0+-3.3% |
| mnv3_tsm_imagenet_frozen | ImageNet frozen | 3.0M | 0.886+-0.006 | 79.1+-5.8% / 16.3+-2.7% | 25.1+-1.2% | 49.0+-1.7% | 45.1+-2.2% |
| mnv3_tsm_imagenet_finetune | ImageNet fine-tuned | 3.0M | 0.870+-0.006 | 73.0+-6.7% / 14.8+-8.5% | 21.7+-1.7% | 45.6+-2.2% | 41.3+-2.1% |
| slowfast_light_scratch | scratch | 1.4M | 0.794+-0.008 | 73.7+-3.6% / 27.6+-3.7% | 14.4+-1.1% | 34.1+-0.9% | 36.0+-2.2% |
| s3d_k400_frozen | Kinetics-400 frozen | 7.9M | 0.784+-0.011 | 63.4+-3.5% / 23.5+-1.4% | 26.9+-0.9% | 51.1+-1.5% | 45.7+-2.0% |
| mc3_18_k400_frozen | Kinetics-400 frozen | 11.5M | 0.868+-0.007 | 75.7+-5.4% / 18.1+-6.0% | 29.0+-1.8% | 54.1+-0.8% | 47.7+-3.4% |
| shufflenet3d_scratch | scratch | 0.4M | 0.811+-0.002 | 75.1+-5.7% / 26.8+-5.9% | 12.1+-2.9% | 32.9+-2.3% | 28.4+-2.9% |
| r2plus1d18_k400_frozen | Kinetics-400 frozen | 31.3M | 0.884+-0.003 | 66.4+-8.3% / 11.5+-4.9% | 23.9+-1.9% | 52.9+-3.2% | 54.3+-0.8% |
| r2plus1d18_k400_finetune | Kinetics-400 fine-tuned | 31.3M | 0.928+-0.005 | 83.6+-4.7% / 9.1+-2.0% | 25.1+-3.2% | 48.9+-0.7% | 46.9+-1.9% |

Reference, deployed PiLENS backbone (frozen MoViNet-A0, Kinetics-600): video AUC 0.919, 13 classes 33%, 4 groups 59%. X3D-S: 0.889, 32%, 61%.

Scripts: `training/backbones_mnv3_tsm_kaggle.py`, `training/backbones_slowfast_s3d_mc3_kaggle.py`, `training/backbones_shufflenet_r2plus1d_kaggle.py`.
