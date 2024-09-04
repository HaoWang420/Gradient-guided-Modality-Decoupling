nnode=4
python \
    train.py \
    -m \
    hydra.job.chdir=False \
    world_size=${nnode} \
    distributed=False \
    mode="save_seg" \
    epochs=600 \
    eval_interval=5 \
    optim=adam \
    optim.lr=8e-4 \
    batch_size=1 \
    test_batch_size=1 \
    model="ensemble" \
    model.output="list" \
    model.feature="False" \
    model.width_ratio="0.5" \
    dataset="brats3d_acn" \
    loss="enumeration" \
    loss.missing_num="2" \
    loss.output="list" \
    workers=6 \
    gpu_ids="'0,1,2,3'" \
    trainer="gmd_trainer" \
    trainer.method="gmd" \
    checkname="gmd-enum-2-adam-acn" \
    resume="results/brats3d-acn/checkpoint.pth.tar"
