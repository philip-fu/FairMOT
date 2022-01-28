cd src
python train.py mot --exp_id ap_yolov5s --data_cfg '../src/lib/cfg/ap.json' --lr 5e-4 --batch_size 16 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64 --gpus 0 --num_epochs 40
cd ..
sudo poweroff
