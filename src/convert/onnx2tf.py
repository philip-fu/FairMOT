import os
import glob
import cv2
import numpy as np
import onnx
from onnx_tf.backend import prepare
#import onnxruntime as ort
#from onnxruntime_extensions import (onnx_op, PyCustomOpDef, get_library_path as _get_library_path)

bz = 3
onnx_model_filename = f"models/dla34conv_864x480_ap_all_ds_25_bz{bz}.onnx"
model = onnx.load(onnx_model_filename)
onnx.checker.check_model(model)
#print(onnx.helper.printable_graph(model.graph))


from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func





tf_rep = prepare(model, strict=True, auto_cast=True)  # prepare tf representation
tf_rep.export_graph(onnx_model_filename.replace('.onnx', ''))  # export the model

""" trying to run inference as onnx
@onnx_op("_DCNv2")
# @tf.func
class _DCNv2(BackendHandler):
    DOMAIN = 'ai.onnx.contrib'

    @classmethod
    def version_1(cls, node, **kwargs):
        return [cls.make_tensor_from_onnx_node(node, **kwargs)]

    @classmethod
    def version_9(cls, node, **kwargs):
        return [cls.make_tensor_from_onnx_node(node, **kwargs)]

    @classmethod
    def version_10(cls, node, **kwargs):
        return [cls.make_tensor_from_onnx_node(node, **kwargs)]


@onnx_op(op_type='_DCNv2', domain='ai.onnx.contrib',
         inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float], outputs=[PyCustomOpDef.dt_float])
def _DCNv2(x, y, z, p):
    return p

so = ort.SessionOptions()
so.register_custom_ops_library(_get_library_path())

ort_session = ort.InferenceSession(model.SerializeToString(), so)
img_files = glob.glob(os.path.join('images/test_ap/*.jpg'))
for img_file in img_files:
    img = cv2.imread(img_file)
    img = cv2.resize(img, (1088, 608))
    img = np.expand_dims(img, axis=0)
    img = np.transpose(img, (0,3,1,2)).astype(np.float32)

    outputs = ort_session.run(
        None,
        {"input.1": img},
    )
    print(outputs)
"""