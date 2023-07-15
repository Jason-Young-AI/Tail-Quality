## HowTo

0. Change into sources directory:
```
$ cd sources
```

1. First of all, convert PyTorch official weights to `*.h5` and `*.tf`, one need to change into PyTorch workload to make the conversion (from PyTorch to NumPy), follow the instructions in [here](../PyTorch/README.md/#weights-conversion-pytorch-to-numpy). Then, assume the `*.npy` is placed in `${BIN_DIR}`, then convert NumPy model weights into TensorFlow format by using the command shown in below:
```
$ python vit.py --mode convert --model-path ${BIN_DIR}/jx_vit_large_p32_tf2named_numpy_state_dict.npy --h5-path ${BIN_DIR}/jx_vit_large_p32.h5 --tf-path ${BIN_DIR}/jx_vit_large_p32.tf
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
- <a href='#local'>Local Mode</a> - Please use the local mode;
- <a href='#serve'>Serve Mode</a> - The performance recorded are wrong, so it is just an example to show the usage of serving.

### <a id='local'>Local Mode</a>
```
$ python run.py --run-number 100 --batch-size 256 --dataset-path ${EXT_DIR} --results-basepath ${RES_DIR}/${RES_NAME} --local --model-path ${BIN_DIR}/jx_vit_large_p32.h5
```

### <a id='serve'>Serve Mode</a>

0. Now you are in `sources` directory, first change into Main directory:
```
$ cd ..
```

2. All settings are set in `configs/*.config`, and serve.sh will read this file:
```
$ ./serve.sh start
```

3. In another terminal, change into sources directory. Suppose `inference_address` is set as `${URL}` and model name is ${MODEL_NAME} run then following command:
```
$ cd sources
$ python run.py --run-number 100 --batch-size 256 --dataset-path ${EXT_DIR} --results-basepath ${RES_DIR}/${RES_NAME} --server-url ${URL} --server-model ${MODEL_NAME}
```

4. If command finished, stop serve:
```
$ ./serve.sh stop
```