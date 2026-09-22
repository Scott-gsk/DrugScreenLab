# Stage2 wave2 launch (DRAFT)

- where: local WSL drugscreening-gpu (authoritative)
- train/test: patient outer 5-fold DRAFT; mcrc_v3; 48 drugs
- GPU0: deeper DualMLP folds 0-2, epochs 60 — pid/log under logs/stage2_deeper_gpu0_20260918_164104.*
- GPU1: deeper DualMLP folds 3-4, epochs 60 — logs/stage2_deeper_gpu1_20260918_164104.*
- CPU: hallmark pathway mean + drug one-hot Ridge — logs/stage2_pathway_ridge_20260918_164104.*
- launched: 20260918_164104
