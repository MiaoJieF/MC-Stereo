python train_stereo_eth3d.py --train_datasets eth3d \
                             --batch_size 4 \
                             --image_size 384 688 \
                             --train_iters 22 \
                             --restore_ckpt checkpoints/sceneflow/mc-stereo_sceneflow.pth \
                             --valid_iters 32 \
                             --spatial_scale -0.2 0.4 \
                             --saturation_range 0 1.4 \
                             --n_downsample 2 \
                             --num_steps 60000