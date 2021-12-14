# Convert to tf model and inference

Only `diaconv_34` is supported and tested at the moment.

```
python src/dladcn_export_onnx.py mot --arch 'dlaconv_34'
python src/onnx2tf.py
python src/export/tf_infer.py
```