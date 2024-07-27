## Quick Run
0. Just Setup Environment with `Conda`:
```
./setup_env.sh
```

1. Setup Datasets MS-COCO:
```
./setup_datasets.sh
```

2. Get Model Weights:
```
./setup_model.sh
```

3. Directly Run Script:
```
./run_tq.sh
```

## HowTo

0. First of all, download DETR model weight from [facebookresearch/detr](https://github.com/facebookresearch/detr) to `${BIN_DIR}`:
    * DETR-DC5-R101 for batch size 1:
    ```
    $ wget -P ${BIN_DIR} https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth
    ```

1. Then, download Backbone(ResNet101) model weights from [torchvision](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html#torchvision.models.ResNet101_Weights) to `${BIN_DIR}`, note that there are two types of pretrained model:
```
$ wget -P ${BIN_DIR} https://download.pytorch.org/models/resnet101-63fe2227.pth
```

2. Combine model weights and backbone weights:
    * DETR-DC5-R101 for batch size 1:
    ```
    $ cd sources/with_dilation_aka_bsz1/
    $ python detr.py --mode create --detr-path ${BIN_DIR}/detr-r101-dc5-a2e86def.pth --backbone-path ${BIN_DIR}/resnet101-63fe2227.pth --model-path ${BIN_DIR}/DETR_ResNet101_BSZ1.pth
    ```
    * DETR-R101 for batch size 2:
    ```
    $ cd sources/without_dilation_aka_bsz2/
    $ python detr.py --mode create --detr-path ${BIN_DIR}/detr-r101-2c7b67e5.pth --backbone-path ${BIN_DIR}/resnet101-63fe2227.pth --model-path ${BIN_DIR}/DETR_ResNet101_BSZ2.pth
    ```

3. Assume the MS-COCO Validation dataset are **extracted** and placed in `${DATA_DIR}`, We expect the directory structure to be the following:
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
- <a href='#local'>Local Mode</a>
- <a href='#serve'>Serve Mode</a>

#### <a id='local'>Local Mode</a>
* DETR-DC5-R101 for batch size 1:
```
$ python run.py --run-number 100 --batch-size 1 --dataset-path ${DATA_DIR} --results-basepath ${RES_DIR}/${RES_NAME} --local --model-path ${BIN_DIR}/DETR_ResNet101_BSZ1.pth
```
* DETR-R101 for batch size 2:
```
$ python run.py --run-number 100 --batch-size 2 --dataset-path ${DATA_DIR} --results-basepath ${RES_DIR}/${RES_NAME} --local --model-path ${BIN_DIR}/DETR_ResNet101_BSZ2.pth
```

#### <a id='serve'>Serve Mode</a>

0. Now you are in `sources/with*bsz*/` directory, first change into Main directory:
```
$ cd ../../
```

1. Archive all the files:
    * DETR-DC5-R101 for batch size 1:
    ```
    $ ./archive.sh BSZ1
    ```
    * DETR-R101 for batch size 2:
    ```
    $ ./archive.sh BSZ2
    ```

2. All settings, such as inference url (`inference_address`) and models' settings (`models`), about serve mode are set in `ts_config_[bsz1|bsz2].properties`, and serve.sh will read this file:
    * DETR-DC5-R101 for batch size 1:
    ```
    $ ./serve.sh start BSZ1
    ```
    * DETR-R101 for batch size 2:
    ```
    $ ./serve.sh start BSZ2
    ```

3. In another terminal, change into sources directory. Suppose `inference_address` is set as `${URL}` and model name is ${MODEL_NAME} run then following command:
    * DETR-DC5-R101 for batch size 1:
    ```
    $ cd sources
    $ python run.py --run-number 100 --batch-size 1 --dataset-path ${EXT_DIR} --results-basepath ${RES_DIR}/${RES_NAME} --server-url ${URL}/predictions/${MODEL_NAME}
    ```
    * DETR-R101 for batch size 2:
    ```
    $ cd sources
    $ python run.py --run-number 100 --batch-size 2 --dataset-path ${EXT_DIR} --results-basepath ${RES_DIR}/${RES_NAME} --server-url ${URL}/predictions/${MODEL_NAME}
    ```

4. If command finished, stop serve:
    * DETR-DC5-R101 for batch size 1:
    ```
    $ ./serve.sh stop BSZ1
    ```
    * DETR-R101 for batch size 2:
    ```
    $ ./serve.sh stop BSZ2
    ```