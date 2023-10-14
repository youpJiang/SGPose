CUDA_VISIBLE_DEVICES=0 python3 run_SGPose.py --note poseaug --posenet_name 'videopose' --lr_p 1e-4 \
--checkpoint './' --keypoints gt --keypoints_target gt \
--dataset_target '3dpw' --pretrain_path '/data/jyp/adaptpose/pretrain_baseline/ckpt_best.pth.tar'  --pad 13 \
--df 6 --tg 5 --debug true --warmup 0 --epochs 50 --attention true --mixtrain true  \
 --decay_epoch 0 --ours2 true --psudo_root false --batch_size 4096