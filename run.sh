CUDA_VISIBLE_DEVICES=3 python3 run_adaptpose.py --note poseaug --posenet_name 'videopose' --lr_p 1e-4 \
--checkpoint '/data2021/jyp/adaptpose/checkpoint/adaptpose' --keypoints gt --keypoints_target gt \
--dataset_target '3dhp' --pretrain_path '/home/jyp/projects/AdaptPose_mine/checkpoint/pretrain_baseline/videopose/gt/3dhp/ckpt_best.pth.tar'  --pad 13 \
--df 6 --tg 5 --debug false --warmup 5 --epochs 50 --attention false --mixtrain true  \
 --decay_epoch 0 --ours2 true --psudo_root false --batch_size 4096