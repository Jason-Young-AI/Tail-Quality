## HowTo

0. First of all, download Vision Transformer model weights from [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) to `${BIN_DIR}` or manually download from [timm/vit_large_patch32_384.orig_in21k_ft_in1k](https://huggingface.co/timm/vit_large_patch32_384.orig_in21k_ft_in1k)
```
$ wget -P ${BIN_DIR} https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth
```

1. Change into sources directory:
```
$ cd sources
```

2. Assume the ImageNet Validation 2012 dataset archive are placed in `${DATA_DIR}`, One could extract the validation dataset into `${EXT_DIR}` by using following command:
```
$ python imagenet.py --save-dir ${EXT_DIR} --dataset-path ${DATA_DIR}
```

### Result Files Format

All results are saved in `json` format.
* `${RES_NAME}.main.*` - record main results about:
    1. `origin_image_size` - list()
    2. `batch_image_size` - list()
    3. `result`
        - `top5_class_indices` - list()
        - `top5_probabilities` - list()
    4. `image_id` - int
* `${RES_NAME}.time.*` - record performance results about:
    1. `inference_time` - float
    2. `postprocess_time` - float
    3. `preprocess_time` - float

Suppose one want to run the workload 100 times, and set batch size as 256, then the following command will save all the results in `${RES_DIR}` and all files will be named with `${RES_NAME}`:
- <a href='#local'>Local Mode</a>
- <a href='#serve'>Serve Mode</a>

#### <a id='local'>Local Mode</a>
```
$ python run.py --run-number 100 --batch-size 256 --dataset-path ${EXT_DIR} --results-basepath ${RES_DIR}/${RES_NAME} --local --model-path ${BIN_DIR}/jx_vit_large_p32_384-9b920ba8.pth
```

#### <a id='serve'>Serve Mode</a>

0. Now you are in `sources` directory, first change into Main directory:
```
$ cd ..
```

1. Archive all the files:
```
$ ./archive.sh
```

2. All settings, such as inference url (`inference_address`) and models' settings (`models`), about serve mode are set in `ts_config.properties`, and serve.sh will read this file:
```
$ ./serve.sh start
```

3. In another terminal, change into sources directory. Suppose `inference_address` is set as `${URL}` and model name is ${MODEL_NAME} run then following command:
```
$ cd sources
$ python run.py --run-number 100 --batch-size 256 --dataset-path ${EXT_DIR} --results-basepath ${RES_DIR}/${RES_NAME} --server-url ${URL}/predictions/${MODEL_NAME}
```

4. If command finished, stop serve:
```
$ ./serve.sh stop
```

### Weights Conversion: PyTorch to NumPy
* NOTE: Your may need to convert the PyTorch weights into NumPy format for converting to the weights of TensorFlow workload, run the following command.
```
$ cd sources
$ python vit.py --model-path ${BIN_DIR}/jx_vit_large_p32_384-9b920ba8.pth --convert --save-path ${BIN_DIR}/jx_vit_large_p32_tf2named_numpy_state_dict.npy
```