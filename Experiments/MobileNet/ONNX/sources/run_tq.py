import mxnet as mx
import numpy as np
from mxnet import gluon, nd
from mxnet import profiler
from mxnet.gluon.data.vision import transforms
from gluoncv.data import imagenet
from collections import namedtuple
from mxnet.contrib.onnx.onnx2mx.import_model import import_model

import os
import gc
import sys
import time
import json
import numpy
import pickle
import pathlib
import logging
import argparse
import matplotlib.pyplot as plt

from pathlib import Path

from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import jensenshannon
from typing import List


from KDEpy.bw_selection import improved_sheather_jones
# from KDEpy import bw_selection 


def set_logger(
    name: str,
    mode: str = 'both',
    level: str = 'INFO',
    logging_filepath: pathlib.Path = None,
    show_setting_log: bool = True
):
    assert mode in {'both', 'file', 'console'}, f'Not Support The Logging Mode - \'{mode}\'.'
    assert level in {'INFO', 'WARN', 'ERROR', 'DEBUG', 'FATAL', 'NOTSET'}, f'Not Support The Logging Level - \'{level}\'.'

    logging_filepath = pathlib.Path(logging_filepath) if isinstance(logging_filepath, str) else logging_filepath

    logging_formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers.clear()

    if mode in {'both', 'file'}:
        if logging_filepath is None:
            logging_dirpath = pathlib.Path(os.getcwd())
            logging_filename = 'younger.log'
            logging_filepath = logging_dirpath.joinpath(logging_filename)
            print(f'Logging filepath is not specified, logging file will be saved in the working directory: \'{logging_dirpath}\', filename: \'{logging_filename}\'')
        else:
            logging_dirpath = logging_filepath.parent
            logging_filename = logging_filepath.name
            logging_filepath = str(logging_filepath)
            print(f'Logging file will be saved in the directory: \'{logging_dirpath}\', filename: \'{logging_filename}\'')

        file_handler = logging.FileHandler(logging_filepath, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging_formatter)
        logger.addHandler(file_handler)

    if mode in {'both', 'console'}:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging_formatter)
        logger.addHandler(console_handler)

    logger.propagate = False

    print(f'Logger: \'{name}\' - \'{mode}\' - \'{level}\'')

    return logger


def kde_aic(bandwidth, ins_times):
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(ins_times)
    log_likelihood = kde.score(ins_times)
    num_params = 2  # KDE has two parameters: bandwidth and kernel
    num_samples = ins_times.shape[0]
    return -2 * log_likelihood + 2 * num_params + (2 * num_params * (num_params + 1)) / (num_samples - num_params - 1)


def gmm_aic(n_components, ins_times):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(ins_times)
    return gmm.aic(ins_times)


def fit(ins_times, fit_type='kde'):
    ins_times = numpy.array(ins_times).reshape(-1, 1)
    if fit_type == 'kde':
        # bandwidth_grid = [0.005, 0.01, 0.03, 0.07, 0.1]
        # best_bandwidth  = min(bandwidth_grid, key=lambda x: kde_aic(x, ins_times))
        best_bandwidth = improved_sheather_jones(ins_times)
        distribution_model = KernelDensity(bandwidth=best_bandwidth).fit(ins_times)
    if fit_type == 'gmm':
        n_components_grid = [2, 3, 4, 5, 6]
        best_n_components = min(n_components_grid, key=lambda x: gmm_aic(x, ins_times))
        distribution_model = GaussianMixture(n_components=best_n_components).fit(ins_times)
    return distribution_model


def check_fit_dynamic(fit_distribution_models, fit_distribution_model, all_times, window_size):
    total_js_dis = 0
    all_times = numpy.array(all_times)
    current_distribution = fit_distribution_model 
    compared_distributions = fit_distribution_models
    for compared_distribution in compared_distributions:
        epsilon = 1e-8
        x = numpy.linspace(all_times.min(), all_times.max(), 1000).reshape(-1, 1) 
        js_dis = jensenshannon(numpy.exp(current_distribution.score_samples(x))+epsilon, numpy.exp(compared_distribution.score_samples(x))+epsilon)
        total_js_dis += js_dis
    avg_jsd = total_js_dis/window_size
    return numpy.sqrt(avg_jsd)


def inference(parameters):
    val_data = parameters['val_data']
    mod = parameters['mod']
    fake_run = parameters['fake_run']
    results_basepath = parameters['results_basepath']
    batch_size = parameters['batch_size']
    ctx = parameters['ctx']

    only_quality = parameters['only_quality']
    golden_path = parameters['golden_path']
    result_path = parameters['result_path']

    tmp_inference_dic = dict()
    tmp_total_dic = dict() 

    overall_result_dic = dict()
    overall_golden_dic = dict()

    predicted_label_top1_list = list()
    predicted_label_top5_list = list()
    labels_list = list()
    if fake_run:
        origin_quality = dict(
            top1_acc = dict(),
            top5_acc = dict(),
        )
    Batch = namedtuple('Batch', ['data'])
    val_data = tqdm(val_data, ascii=True)

    # profiler.set_state('run')
    a = time.perf_counter()
    for batch_id, batch in enumerate(val_data, start=1):
        datas = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

        inference_start = time.perf_counter()
        preprocess_time = inference_start - a 
        mod.forward(Batch([datas[0]]))
        outputs=mod.get_outputs()
        inference_end = time.perf_counter()
        inference_time = inference_end - inference_start
        tmp_inference_dic[batch_id] = float(inference_time)

        postprocess_start = time.perf_counter()
        b = time.perf_counter()
        for i, output in enumerate(outputs[0]):
            if i == 0:
                c = time.perf_counter()
                # mx.nd.waitall()
                # profiler.set_state('stop') 
                fake1 = mx.nd.argmax(output).asnumpy() 
                fake2 = mx.nd.topk(output).asnumpy()
                # profiler.set_state('run')
                postprocess_start = time.perf_counter() + c - b  
            predicted_label_top1_list.append(mx.nd.argmax(output).asnumpy())
            predicted_label_top5_list.append(mx.nd.topk(output, k=5).asnumpy())

        postprocess_end = time.perf_counter()
        postprocess_time =  postprocess_end - postprocess_start

        total_time = preprocess_time + inference_time + postprocess_time
        tmp_total_dic[batch_id] = float(total_time)
        # print('preprocess_time', preprocess_time)
        # print('inference_time', inference_time)
        # print('postprocess_time', postprocess_time)
        # print('total_time', total_time)

        if fake_run:
            for label in labels[0]:
                labels_list.append(label)
            batch_acc1, batch_acc5 = accuracy(predicted_label_top5_list[-len(outputs[0]):], labels_list[-len(outputs[0]):])
            origin_quality['top1_acc'][batch_id] = batch_acc1 
            origin_quality['top5_acc'][batch_id] = batch_acc5 
            if only_quality:
                overall_result_dic[batch_id] = [([op1.tolist(), top5.tolist()) for top1, top5 in zip(predicted_label_top1_list[-len(outputs[0]):], predicted_label_top5_list[-len(outputs[0]):])]
                overall_golden_dic[batch_id] = [label for label in labels[0]]
        del outputs
        del datas
        del labels
        # mx.nd.waitall()
        # profiler.set_state('stop') 
        a = time.perf_counter()

    if fake_run:
        acc1, acc5 = accuracy(predicted_label_top5_list, labels_list)
        print('top1-acc and top5-acc : ', acc1, acc5) # acc in the whole val set
        with open(results_basepath.joinpath('Origin_Quality.json'), 'w') as f:
            json.dump(origin_quality, f, indent=2)
        if only_quality:
            with open(result_path, 'wb') as result_file:
                pickle.dump(overall_result_dic, result_file)
            with open(golden_path, 'wb') as golden_file:
                pickle.dump(overall_golden_dic, golden_file)

    del predicted_label_top1_list 
    del predicted_label_top5_list 
    gc.collect()

    return tmp_inference_dic, tmp_total_dic


def accuracy(predicted_label_top5_list, labels):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    top1_acc = 0
    top5_acc =0
    for i, (predicted_label_top5, label) in enumerate(zip(predicted_label_top5_list, labels)):
        if predicted_label_top5[0] == label:
            top1_acc += 1
        if np.isin(label, predicted_label_top5):
            top5_acc += 1
    top1_acc = round(100 * (top1_acc / len(labels)), 6) 
    top5_acc = round(100 * (top5_acc / len(labels)), 6)
    return top1_acc,top5_acc


def draw_rjsds(rjsds: List, results_basepath: pathlib.Path):
    inference_data = list(range(1, len(rjsds['inference']) + 1))
    total_data = list(range(1, len(rjsds['total']) + 1))
    fig, ax = plt.subplots()
    
    ax.plot(inference_data, rjsds['inference'], marker='o', linestyle='-', color='b', label='rJSD(inference time)')
    ax.plot(total_data, rjsds['total'], marker='o', linestyle='-', color='y', label='rJSD(total time)')
    ax.set_title('rJSD Fitting Progress')
    ax.set_xlabel('Fitting Round')
    
    ax.set_ylabel('rJSD')
    ax.grid(True)
    ax.legend()
    plt.savefig(results_basepath.joinpath("rJSDs.jpg"), format="jpg")
    plt.savefig(results_basepath.joinpath("rJSDs.pdf"), format="pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multiple times on the Whole Test Dataset")
    parser.add_argument('--results-basepath', type=str, required=True)
    parser.add_argument('--warm-run', type=int, default=1)
    parser.add_argument('--fake-run', type=bool, default=True) # To avoid the outliers processed during the first inference
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--fit-run-number', type=int, default=2)
    parser.add_argument('--rJSD-threshold', type=float, default=0.05)
    parser.add_argument('--max-run', type=int, default=0)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)

    parser.add_argument('--only-quality', action='store_true')
    parser.add_argument('--golden-path', type=str)
    parser.add_argument('--result-path', type=str)
    parser.add_argument('--others-path', type=str)

    args = parser.parse_args()

    if args.only_quality:
        assert args.golden_path is not None
        assert args.result_path is not None
        assert args.others_path is not None

    model_path = Path(args.model_path)
    assert model_path.is_file(), f"Model Weights path {model_path.absolute()} does not exist."

    dataset_root = Path(args.dataset_path)
    assert dataset_root.is_dir(), f"provided Imagenet path {dataset_root} does not exist"
    if args.device == 'gpu':
        ctx = [mx.gpu(0)]
    else:
        ctx = [mx.cpu()]    
    batch_size = args.batch_size
    sym, arg_params, aux_params = import_model(model_path)
    num_workers = 8
    # Define image transforms
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    # Load and process input
    val_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(dataset_root, train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # Load module
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
            label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    results_basepath = pathlib.Path(args.results_basepath)
    warm_run = args.warm_run 
    window_size = args.window_size
    fit_run_number = args.fit_run_number
    rJSD_threshold = args.rJSD_threshold
    fake_run = args.fake_run
    max_run = args.max_run

    result_path = results_basepath.joinpath('All_Times.pickle')
    rjsds_path = results_basepath.joinpath('All_rJSDs.pickle')
    fit_distribution_dir = results_basepath.joinpath('All_PDFs')
    if not fit_distribution_dir.exists():
        fit_distribution_dir.mkdir(parents=True, exist_ok=True)
    fit_distribution_model_paths = list(fit_distribution_dir.iterdir())
    fit_distribution_number = len(fit_distribution_model_paths)//2

    logger = set_logger(name='Tail-Quality', mode='both', level='INFO', logging_filepath=results_basepath.joinpath('Tail-Quality.log'))
    total_batches = 0
    if result_path.exists():
        with open (result_path, 'rb') as f:
            results = pickle.load(f)
        already_run = len(results['inference'])
        del results
    else:
        already_run = 0
    sucess_flag = False
    loop = 0 # for debugging
    while not sucess_flag:
        loop += 1 # for debugging
        params = {
            'val_data': val_data,
            'mod': mod, 
            'fake_run': fake_run,
            'results_basepath': results_basepath,
            'batch_size': batch_size,
            'ctx': ctx,
            'only_quality': args.only_quality,
            'golden_path': args.golden_path,
            'result_path': args.result_path,
            'others_path': args.others_path,
        }

        logger.info(f'-------before loop {loop}-------')
        logger.info(f'already_run: {already_run}')
        logger.info(f'warm_run: {warm_run}')
        logger.info(f'fit_distribution_number: {fit_distribution_number}')

        # if not fake_run:
        #     profiler.set_config(profile_all=True, aggregate_stats=True, filename='profile_output.json')
        
        tmp_inference_dic, tmp_total_dic = inference(params)
        if args.only_quality:
            logger.info(f'Only Get Quality')
            break
        # if not fake_run:
        #     mx.nd.waitall()
        #     profiler.set_state('stop')
        #     logger.info(profiler.dumps())

        logger.info(f'after inference')
        if not fake_run:
            already_run += 1 
            logger.info(f'already_run: {already_run}')  
            all_inference_times = list()
            all_total_times = list() 
            if result_path.exists(): 
                with open (result_path, 'rb') as f:
                    results = pickle.load(f) 
                    tmp_results = results.copy()
                    for inference_times in tmp_results['inference']:
                        for inference_time in inference_times.values():
                            all_inference_times.append(inference_time)
                    for total_times in tmp_results['total']:
                        for total_time in total_times.values():
                            all_total_times.append(total_time)
                    del results
                tmp_results['inference'].append(tmp_inference_dic)
                tmp_results['total'].append(tmp_total_dic)
            else:
                tmp_results = dict(
                    inference = list(),
                    total = list()
                )
                tmp_results['inference'].append(tmp_inference_dic)
                tmp_results['total'].append(tmp_total_dic)

            for key, value in tmp_inference_dic.items():
                all_inference_times.append(value)
            for key, value in tmp_total_dic.items():
                all_total_times.append(value)

            logger.info(f'(already_run - warm_run) % fit_run_number == {(already_run - warm_run) % fit_run_number}') 
            logger.info(f"fit_distribution_number % window_size == {fit_distribution_number % window_size}")
            if already_run > warm_run and (already_run - warm_run) % fit_run_number == 0:
                fit_inference_distribution_model = fit(all_inference_times) 
                fit_total_distribution_model = fit(all_total_times)
                if fit_distribution_number % window_size == 0 and fit_distribution_number != 0:
                    inference_model_paths = sorted([f for f in fit_distribution_dir.iterdir() if f.stem.split('-')[-2] == 'inference'], key=lambda x: int(x.stem.split('-')[-1]))
                    total_model_paths = sorted([f for f in fit_distribution_dir.iterdir() if f.stem.split('-')[-2] == 'total'], key=lambda x: int(x.stem.split('-')[-1]))
                    fit_inference_distribution_models = list()
                    fit_total_distribution_models = list() 
                    for inference_model_path in inference_model_paths[-window_size:]:
                        with open(inference_model_path, 'rb') as f:
                            distribution_model = pickle.load(f)
                            fit_inference_distribution_models.append(distribution_model) 
                    for total_model_path in total_model_paths[-window_size:]:
                        with open(total_model_path, 'rb') as f:
                            distribution_model = pickle.load(f)
                            fit_total_distribution_models.append(distribution_model)
                            
                    logger.info(f'start_check_fit')
                    inference_rjsd = check_fit_dynamic(fit_inference_distribution_models, fit_inference_distribution_model, all_inference_times, window_size)
                    total_rjsd = check_fit_dynamic(fit_total_distribution_models, fit_total_distribution_model, all_total_times, window_size)
                    logger.info(f'end_check_fit')
                    del fit_inference_distribution_models
                    del fit_total_distribution_models

                    logger.info(f'inference_rjsd is {inference_rjsd} / total_rjsd is {total_rjsd}')
                    sucess_flag = True if inference_rjsd <= rJSD_threshold and total_rjsd <= rJSD_threshold else False
                    if inference_rjsd <= rJSD_threshold:
                        logger.info('inference_times has fitted') 
                    if total_rjsd <= rJSD_threshold:
                        logger.info('total_times has fitted') 
                    logger.info(f'start_draw_rjsds')
                    if rjsds_path.exists():
                        with open(rjsds_path, 'rb') as f:
                            rjsds = pickle.load(f)
                            tmp_rjsds = rjsds.copy()
                            del rjsds
                        tmp_rjsds['inference'].append(inference_rjsd)
                        tmp_rjsds['total'].append(total_rjsd)
                    else:
                        tmp_rjsds = dict(
                            inference = list(),
                            total = list()
                        )
                        tmp_rjsds['inference'].append(inference_rjsd)
                        tmp_rjsds['total'].append(total_rjsd)
                    with open(rjsds_path, 'wb') as f:
                        pickle.dump(tmp_rjsds, f)
                    draw_rjsds(tmp_rjsds, results_basepath) 
                    del tmp_rjsds
                    logger.info(f'end_draw_rjsds')

            
                fit_distribution_number += 1
                with open(fit_distribution_dir.joinpath(f'inference-{fit_distribution_number}.pickle'), 'wb') as f:
                    pickle.dump(fit_inference_distribution_model, f)
                with open(fit_distribution_dir.joinpath(f'total-{fit_distribution_number}.pickle'), 'wb') as f:
                    pickle.dump(fit_total_distribution_model, f)
                del fit_inference_distribution_model
                del fit_total_distribution_model 
                
            with open(result_path, 'wb') as f:
                pickle.dump(tmp_results, f)
                logger.info(f'len tmp_results["inference"] is {len(tmp_results["inference"])}')
            with open(f'{result_path}.json', 'w') as f:
                json.dump(tmp_results, f, indent=2) 
            del tmp_results
            del all_total_times
            del all_inference_times

        
        logger.info(f'-------after loop {loop}-------')
        logger.info(f'already_run: {already_run}')
        logger.info(f'warm_run: {warm_run}')
        logger.info(f'fit_distribution_number: {fit_distribution_number}')
        if fake_run:
            logger.info(f'this run is fake')
        fake_run = False

        if already_run == max_run:
            break 
