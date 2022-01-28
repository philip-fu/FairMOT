cd src
python train.py mot --exp_id ap_all_ds_yolov5s --data_cfg '../src/lib/cfg/ap_all.json' --gpus 0,1,2,3 --num_epochs 25 --lr 5e-4 --batch_size 16 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 && sudo poweroff
