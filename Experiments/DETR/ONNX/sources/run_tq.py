import os
import sys
import time
import json
import numpy
import pickle
import pathlib
import logging
import argparse
import onnxruntime
import matplotlib.pyplot as plt

from pathlib import Path

import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transforms import Compose, DetermineResize, ToTensor, Normalize
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import jensenshannon
from utils import prepare_for_coco_detection
from box_ops import box_cxcywh_to_xyxy
from typing import List
from pycocotools.coco import COCO

from misc import nested_tensor_from_tensor_list


import time
import torch

from KDEpy.bw_selection import improved_sheather_jones


image_preprocessing_local = Compose([
    DetermineResize(800, max_size=1333),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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


def get_model_parameters_number(model: torch.nn.Module) -> int:
    parameters_number = dict()
    for name, parameters in model.named_parameters():
        root_name = name.split('.')[0]
        if root_name in parameters_number:
            parameters_number[root_name] += parameters.numel()
        else:
            parameters_number[root_name] = parameters.numel()

    return parameters_number


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


def load_dataset(anno_path, image_dir):
    coco = COCO(anno_path)
    image_ids = list(sorted(coco.imgs.keys()))
    img_paths = list()

    for image_id in image_ids:
        image_name = coco.loadImgs(image_id)[0]["file_name"]
        img_paths.append(os.path.join(image_dir, image_name))

    return img_paths, image_ids


def postprocess(outputs, image_sizes, image_indices):
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)

    # convert to [x0, y0, x1, y1] format
    boxes = box_cxcywh_to_xyxy(out_bbox)
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = image_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    scale_fct = scale_fct.to(boxes.device)
    boxes = boxes * scale_fct[:, None, :]

    raw_results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

    results = prepare_for_coco_detection(raw_results, image_indices)
    return results


def add_other(coco_results, image_sizes, image_shape):
    coco_results_with_other = list()
    for coco_result, image_size in zip(coco_results, image_sizes):
        coco_results_with_other.append(
            dict(
                origin_image_size=image_size,
                batch_image_size=image_shape,
                result=coco_result,
            )
        )

    return coco_results_with_other


def run_onnx(images, session):
    # Plan A
    # binding = session.io_binding()
    # images = images.contiguous()
    # # print(images.shape)
    # binding.bind_input('input_image', element_type=numpy.float32, device_id=0, shape=tuple(images.shape), device_type='cuda', buffer_ptr=images.data_ptr())
    # pred_logits_shape = (1,100,92)
    # pred_boxes_shape = (1,100,4)
    # pred_logits = torch.empty(pred_logits_shape, dtype=torch.float32, device='cuda:0').contiguous()
    # pred_boxes = torch.empty(pred_boxes_shape, dtype=torch.float32, device='cuda:0').contiguous()
    # binding.bind_output('pred_logits', element_type=numpy.float32, shape=pred_logits_shape, device_type='cuda', buffer_ptr=pred_logits.data_ptr())
    # binding.bind_output('pred_boxes', element_type=numpy.float32, shape=pred_boxes_shape, device_type='cuda', buffer_ptr=pred_boxes.data_ptr())
    # session.run_with_iobinding(binding)

    # Plan B
    pred_logits, pred_boxes = session.run(['pred_logits', 'pred_boxes'], {'input_image': images})

    results = dict(
        pred_logits = torch.from_numpy(pred_logits),
        pred_boxes =  torch.from_numpy(pred_boxes),
    )
    return results


def inference(parameters):
    img_ids = parameters['img_ids']
    img_paths = parameters['img_paths']
    session = parameters['session']
    fake_run = parameters['fake_run']
    results_basepath = parameters['results_basepath']
    assert len(img_paths) == len(img_ids), "Fatal Error!"

    tmp_inference_dic = dict()
    tmp_total_dic = dict()
    all_results = list()
    a = time.perf_counter()
    for batch_id, (img_path, img_id) in tqdm(enumerate(zip(img_paths, img_ids), start=1), ascii=True, total=len(img_ids)):
        batch = [(img_path, img_id), ]
        images = list()
        image_sizes = list()
        image_indices = list()

        for index, (img_path, img_id) in enumerate(batch):
            raw_image = Image.open(img_path).convert("RGB")

            w, h = raw_image.size

            image = image_preprocessing_local(raw_image)
            images.append(image)
            image_sizes.append(torch.as_tensor([int(h), int(w)]))
            image_indices.append(img_id)

        # images = nested_tensor_from_tensor_list(images).to('cuda:0')
        images = numpy.array(images)
        # images = torch.from_numpy(numpy.array(images))
        # images = torch.from_numpy(numpy.array(images)).to('cuda:0')
        # image_sizes = torch.stack(image_sizes, dim=0).to('cuda:0')
        image_sizes = torch.stack(image_sizes, dim=0)

        inference_start = time.perf_counter()
        preprocess_time = inference_start - a 
        results = run_onnx(images, session)
        inference_end = time.perf_counter()
        inference_time = inference_end - inference_start
        tmp_inference_dic[batch_id] = float(inference_time)

        postprocess_start = time.perf_counter()
        results = postprocess(results, image_sizes, image_indices)
        # print(results)
        postprocess_end = time.perf_counter()
        postprocess_time = postprocess_end - postprocess_start
        total_time = preprocess_time + inference_time + postprocess_time
        # print(inference_time)
        # print(total_time)
        tmp_total_dic[batch_id] = float(total_time)

        if fake_run:
            image_sizes = image_sizes.tolist()
            batch_image_shape = images.shape[-2:]
            all_results.append(add_other(results, image_sizes, batch_image_shape))

        # logger.info('total_time, inference_time: ', total_time, inference_time)
        a = time.perf_counter()

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
    parser.add_argument('--min-run', type=int, required=False)
    parser.add_argument('--results-basepath', type=str, required=True)
    parser.add_argument('--warm-run', type=int, default=1)
    parser.add_argument('--fake-run', type=bool, default=True) # To avoid the outliers processed during the first inference
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--fit-run-number', type=int, default=2)
    parser.add_argument('--rJSD-threshold', type=float, default=0.05)
    parser.add_argument('--max-run', type=int, default=0)

    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)

    args = parser.parse_args()

    model_path = Path(args.model_path)
    assert model_path.is_file(), f"Model Weights path {model_path.absolute()} does not exist."

    dataset_root = Path(args.dataset_path)
    assert dataset_root.is_dir(), f"provided COCO path {dataset_root} does not exist"

    anno_path = os.path.join(dataset_root, "annotations", "instances_val2017.json")
    image_dir = os.path.join(dataset_root, "val2017")

    img_paths, img_ids = load_dataset(anno_path, image_dir)

    results_basepath = pathlib.Path(args.results_basepath)
    min_run = args.min_run 
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

    # [!Begin] Model Initialization
    # options = onnxruntime.SessionOptions()
    # options.enable_profiling=True
    # session = onnxruntime.InferenceSession(model_path,sess_options=options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']) 
    
    # X is numpy array on cpu

    session_options = onnxruntime.SessionOptions()
    # session_options.enable_profiling = True 
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # onnxruntime.set_default_logger_severity(1) 
    session = onnxruntime.InferenceSession(
            model_path,
            session_options=session_options,
            providers=['CPUExecutionProvider']
    )

    # [!End] Model Initialization

    sucess_flag = False
    loop = 0 # for debugging
    with torch.no_grad():
        while not sucess_flag:
            loop += 1 # for debugging
            params = {
                'img_paths': img_paths,
                'img_ids': img_ids,
                'session': session, 
                'fake_run': fake_run,
                'results_basepath': results_basepath,
            }

            logger.info(f'-------before loop {loop}-------')
            logger.info(f'already_run: {already_run}')
            logger.info(f'warm_run: {warm_run}')
            logger.info(f'fit_distribution_number: {fit_distribution_number}')

            tmp_inference_dic, tmp_total_dic = inference(params)
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
                if (already_run - warm_run) % fit_run_number == 0 and already_run != warm_run:
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
            

            
    
