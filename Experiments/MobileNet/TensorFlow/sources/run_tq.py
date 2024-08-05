import time
from collections import namedtuple
import tensorflow.compat.v1 as tf
from mobilenet.dataset import ImageNet
from mobilenet.preprocessing import transform

import os
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

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import jensenshannon
from typing import List

import time

from KDEpy.bw_selection import improved_sheather_jones


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
    next_batch = parameters['next_batch']
    fake_run = parameters['fake_run']
    results_basepath = parameters['results_basepath']
    inputs = parameters['inputs']
    predictions = parameters['predictions']
    session = parameters['session']

    only_quality = parameters['only_quality']
    golden_path = parameters['golden_path']
    result_path = parameters['result_path']

    golden_label_list = list()
    predicted_label_top1_list = list()
    predicted_label_top5_list = list()

    tmp_inference_dic = dict()
    tmp_total_dic = dict()

    overall_result_dic = dict()
    overall_golden_dic = dict()

    a = time.perf_counter()
    batch_id = 0
    while True:
        try:
            image_batch, label_batch = session.run(next_batch)
            batch_id += 1
            # label_batch = session.run(next_label_batch)
            inference_start = time.perf_counter()
            preprocess_time = inference_start - a 
            predicted_labels: numpy.ndarray = predictions.eval(feed_dict={inputs: image_batch})
            inference_end = time.perf_counter()
            inference_time = inference_end - inference_start
            # print('INF', inference_time)
            tmp_inference_dic[batch_id] = float(inference_time)
            postprocess_start = time.perf_counter()
            for predicted_label in predicted_labels:
                predicted_label_top1_list.append(predicted_label.argmax())
                predicted_label_top5_list.append(list(predicted_label.argsort()[-5:][::-1]))
            postprocess_end = time.perf_counter()
            postprocess_time =  postprocess_end - postprocess_start
            total_time = preprocess_time + inference_time + postprocess_time
            # print('TOT', total_time)
            tmp_total_dic[batch_id] = float(total_time)
            if fake_run:
                for label in label_batch:
                    golden_label_list.append(int(label))
                if only_quality:
                    overall_result_dic[batch_id] = [([top1], top5) for top1, top5 in zip(predicted_label_top1_list[-len(predicted_labels):], predicted_label_top5_list[-len(predicted_labels):])]
                    overall_golden_dic[batch_id] = golden_label_list
        except tf.errors.OutOfRangeError:
            break
        a = time.perf_counter()

    if fake_run:
        top1_correct = 0
        top5_correct = 0
        total = 0
        for predicted_top1, predicted_top5, golden in zip(predicted_label_top1_list, predicted_label_top5_list, golden_label_list):
            total += 1
            if predicted_top1 == golden:
                top1_correct += 1
            if golden in predicted_top5:
                top5_correct += 1
        print(f"in fake run: Top-1 accuracy: {(top1_correct/total):.5f}, Top-5 accuracy: {(top5_correct/total):.5f}")
        main_results = dict(
            t1acc = top1_correct/total,
            t5acc = top5_correct/total,
            top1 = [int(pt1) for pt1 in predicted_label_top1_list],
            top5 = [[int(pt5i) for pt5i in pt5] for pt5 in predicted_label_top5_list],
            gold = golden_label_list,
        )
        with open(results_basepath.joinpath('Origin_Quality.json'), 'w') as f:
            json.dump(main_results, f, indent=2)
        if only_quality:
            with open(result_path, 'w') as result_file:
                json.dump(overall_result_dic, result_file, indent=2)
            with open(golden_path, 'w') as golden_file:
                json.dump(overall_golden_dic, golden_file, indent=2)

    return  tmp_inference_dic, tmp_total_dic


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
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--workers', default=8, type=int)

    parser.add_argument('--only-quality', action='store_true')
    parser.add_argument('--golden-path', type=str)
    parser.add_argument('--result-path', type=str)

    args = parser.parse_args()

    model_path = Path(args.model_path)
    assert model_path.is_file(), f"Model Weights path {model_path.absolute()} does not exist."

    dataset_root = Path(args.dataset_path)
    assert dataset_root.is_dir(), f"provided Imagenet path {dataset_root} does not exist"

    batch_size = args.batch_size

    tf.compat.v1.disable_eager_execution()
    tf.reset_default_graph()

    graphdef = tf.GraphDef.FromString(open(model_path, 'rb').read())
    inputs, predictions = tf.import_graph_def(graphdef,  return_elements = ['input:0', 'MobilenetV2/Predictions/Reshape_1:0'])

    imagenet = ImageNet(dataset_root.joinpath('val'))
    img_paths = [img_path for img_path, _ in imagenet]
    label_ids = [label_id for _, label_id in imagenet]

    image_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(label_ids, tf.int64))

    instance_ds = tf.data.Dataset.zip((image_ds, label_ds))
    #instance_ds = instance_ds.map(transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    instance_ds = instance_ds.map(transform, num_parallel_calls=args.workers)
    instance_ds = instance_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    instance_it = tf.compat.v1.data.make_initializable_iterator(instance_ds)
    next_batch = instance_it.get_next()

    # image_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    # image_ds = image_ds.map(transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # image_ds = image_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    # image_it = tf.compat.v1.data.make_initializable_iterator(image_ds)
    # next_image_batch = image_it.get_next()

    # label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(label_ids, tf.int64))
    # label_ds = label_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    # label_it = tf.compat.v1.data.make_initializable_iterator(label_ds)
    # next_label_batch = label_it.get_next()

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
    with tf.Session(graph=inputs.graph) as session:
        while not sucess_flag:
            session.run(instance_it.initializer)
            loop += 1 # for debugging
            params = {
                'next_batch': next_batch,
                'fake_run': fake_run,
                'results_basepath': results_basepath,
                'inputs': inputs,
                'predictions': predictions,
                'session': session,
                'only_quality': args.only_quality,
                'golden_path': args.golden_path,
                'result_path': args.result_path,
            }

            logger.info(f'-------before loop {loop}-------')
            logger.info(f'already_run: {already_run}')
            logger.info(f'warm_run: {warm_run}')
            logger.info(f'fit_distribution_number: {fit_distribution_number}')

            tmp_inference_dic, tmp_total_dic = inference(params)
            if args.only_quality:
                logger.info(f'Only Get Quality')
                break
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