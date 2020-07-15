CUDA_VISIBLE_DEVICES='0,1' \
    python3 train_parall_refactor.py \
        --network r18 \
        --dataset gan_emore \
        --loss feat_mom_phi \
        --psi-norm-name penalize_with_cos_psi \
        --add-gan-loss --restore-scale 1.0 --gan-mom 0.0 \
        --verbose 200


