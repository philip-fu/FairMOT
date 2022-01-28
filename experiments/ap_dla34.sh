cd src
python train.py mot --exp_id ap_dla34 --data_cfg '../src/lib/cfg/ap.json' --load_model '../exp/mot/ap_dla34/model_last.pth' --gpus 0,1 --num_epochs 41 --resume && sudo poweroff
cd ..
#sudo poweroff
