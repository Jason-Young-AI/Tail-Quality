## HowTo

0. First of all, manually download converted DETR model weights from [Leonardo-Blanger/detr_tensorflow](https://github.com/Leonardo-Blanger/detr_tensorflow)([Google Drive](https://drive.google.com/drive/folders/1OMzJNxsx-D5lyLgrQokLvbpvrZ5rM9rW)) to `${BIN_DIR}`, note that there are two types of pretrained model:
    * DETR-DC5-R101 for batch size 1: detr-r101-dc5-a2e86def.h5
    * DETR-R101 for batch size 2: detr-r101-2c7b67e5.h5

1. One may want to know how to convert PyTorch weights to TensorFlow, please refer to [this issue](https://github.com/Leonardo-Blanger/detr_tensorflow/issues/2#issuecomment-730008815), below is a snapshot:
    0. Env
    ```
    pip install onnx onnx_tf tensorflow_probability
    ```
    1. Convert pytorch model to onnx format: load your pytorch detr model and use .eval() to convert it to an inference model. Then create a dummy variable input and feed it to the export() method. Make sure to set `opset_version=11`
    ```
    dummy_input = Variable(torch.randn(1, 3, 800, 1777)).to('cuda:0')

    # you may change the names
    input_names = ['input_image']
    output_names = ['pred_logits', 'pred_boxes']

    # I use dynamic axes to make the model accept a variable batch size, width, and height for the input images
    dynamic_axes = {'input_image': {0: 'batch_size', 2: 'width', 3: 'height'}}

    torch.onnx.export(model, dummy_input, "model.onnx", input_names=input_names, output_names=output_names,
                    dynamic_axes=dynamic_axes, verbose=1, opset_version=11)
    ```
    2. Convert onnx to tf saved_model (for tf serving):
    ```
    import onnx
    from onnx_tf.backend import prepare

    model = onnx.load('model.onnx')

    # Import the ONNX model to Tensorflow
    tf_rep = prepare(model)

    # don't use the ".pb" in the name of the exported file, so that it creates a proper folder for the weights
    tf_rep.export_graph("detr")
    ```

2. Assume the MS-COCO Validation dataset are **extracted** and placed in `${DATA_DIR}`, We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  val2017/      # val images
```

### Result Files Format
All results are saved in `json` format.
* `${RES_NAME}.main.*` - record main results about:
    1. `origin_image_size` - list()
    2. `batch_image_size` - list()
    3. `result`
        - `image_id` - int
        - `category_id` - int
        - `bbox` - list()
        - `score` - float
* `${RES_NAME}.main_extended.*` - record extended main results for convenience of the evaluation by using pycocotools;
* `${RES_NAME}.time.*` - record performance results about:
    1. `inference_time` - float
    2. `postprocess_time` - float
    3. `preprocess_time` - float

Suppose one want to run the workload 100 times, and set batch size as 1 (DETR-DC5-R101) or 2 (DETR-R101), then the following command will save all the results in `${RES_DIR}` and all files will be named with `${RES_NAME}`:
- <a href='#local'>Local Mode</a> - Please use this mode.
- <a href='#serve'>Serve Mode</a> - Serve mode are not implemented.

#### <a id='local'>Local Mode</a>
* DETR-DC5-R101 for batch size 1:
```
$ python run.py --run-number 100 --batch-size 1 --dataset-path ${DATA_DIR} --results-basepath ${RES_DIR}/${RES_NAME} --local --model-path ${BIN_DIR}/detr-r101-dc5-a2e86def.h5
```
* DETR-R101 for batch size 2:
```
$ python run.py --run-number 100 --batch-size 2 --dataset-path ${DATA_DIR} --results-basepath ${RES_DIR}/${RES_NAME} --local --model-path ${BIN_DIR}/detr-r101-2c7b67e5.h5
```

#### <a id='serve'>Serve Mode</a>
NOTE: No Implementation.