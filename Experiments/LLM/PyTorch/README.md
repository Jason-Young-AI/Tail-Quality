## HowTo

0. First of all, download MMLU dataset from [hendrycks/test](https://github.com/hendrycks/test) to `${DATA_DIR}`:
```
$ wget -P ${DATA_DIR} https://people.eecs.berkeley.edu/~hendrycks/data.tar
```

1. Then, download LLM model weights(Vicuna-13B-v1.3) from [lmsys/vicuna-13b-v1.3](https://huggingface.co/lmsys/vicuna-13b-v1.3) to `${BIN_DIR}`, or just run the run.sh command which will download pretrained weights automaticly from HuggingFace:

2. Assume the MMLU dataset are **extracted** and placed in `${DATA_DIR}`, We expect the directory structure to be the following:
```
path/to/mmlu/
  dev/  # dev tasks
  val/  # val tasks
  test/ # test tasks
```

### Result Files Format
All results are saved in `json` format.
* `${RES_NAME}.main.*` - record main results about:
    1. `task_name`:
        - `pred_answers` - list()
        - `gold_answers` - list()
        - `token_lengths` - int
        - `question_numbers` - int
* `${RES_NAME}.time.*` - record performance results about:
    1. `pred_times`:
        -. `inference_time` - float
        -. `postprocess_time` - float
        -. `preprocess_time` - float

Suppose one want to run the workload 100 times, and set batch size as 1, then the following command will save all the results in `${RES_DIR}` and all files will be named with `${RES_NAME}`:
- <a href='#local'>Local Mode</a>
- <a href='#serve'>Serve Mode</a>

#### <a id='local'>Local Mode</a>
```
$ python run.py --run-number 100 --batch-size 1 --dataset-path ${DATA_DIR} --results-basepath ${RES_DIR}/${RES_NAME} --local --model-path lmsys/vicuna-13b-v1.3
```

#### <a id='serve'>Serve Mode</a>

0. Not Implemented