# changed in src/lib/opts.py
cd src
python train.py mot --exp_id ap_all_ds_dla34conv_860x480 --data_cfg '../src/lib/cfg/ap_all.json' --gpus 0,1,2,3 --num_epochs 25 --arch 'dlaconv_34' --lr 0.0002 --batch_size 24 --input_h 864 --input_res 864 --input_w 480 && sudo poweroff
