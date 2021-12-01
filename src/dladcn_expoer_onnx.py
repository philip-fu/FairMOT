import os
import numpy as np
import torch
import torch.onnx.utils as onnx
import lib.models.networks.pose_dla_dcn as net

from lib.models.model import create_model, load_model
from lib.opts import opts
from collections import OrderedDict
import cv2

opt = opts().init()
model = create_model(opt.arch, opt.heads, opt.head_conv, pretrain=False)
model_state_dict = model.state_dict()
model_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/dla34_ap_all_ds_20.pth")
print(f"Loading {model_filename}")
checkpoint = torch.load(model_filename, map_location="cpu")
checkpoint = checkpoint["state_dict"]
change = OrderedDict()
for key, op in checkpoint.items():
    change[key.replace("module.", "", 1)] = op

msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'

for k in change:
    if k not in model_state_dict:
        print('Drop parameter {}.'.format(k) + msg)

for k in model_state_dict:
    if not (k in change):
        print('No param {}.'.format(k) + msg)
        change[k] = model_state_dict[k]

model.load_state_dict(change, strict=False)
model.eval()
model.cuda()

input = torch.zeros((1, 3, 608, 1088)).cuda()    # the size could be reset

onnx.export(model, (input), model_filename.replace('pth', '.onnx'), output_names=["hm", "wh", "reg", "hm_pool"], verbose=False, operator_export_type=onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
