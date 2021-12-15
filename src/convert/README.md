# Convert to tf model and inference

Only `dlaconv_34` is supported and tested at the moment.

```
python src/dladcn_export_onnx.py mot --arch 'dlaconv_34'
python src/convert/onnx2tf.py
python src/convert/tf_infer.py
```