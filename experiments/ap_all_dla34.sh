cd src
python train.py mot --exp_id ap_all_ds_dla34 --data_cfg '../src/lib/cfg/ap_all.json' --load_model '../exp/mot/ap_all_dla34/model_last.pth' --gpus 0,1,2,3 --num_epochs 25 && sudo poweroff