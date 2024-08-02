from cgi import parse_multipart
import enum
import os
import sys
import time
import numpy
import json
import pickle
import pathlib
import logging
import warnings
import argparse

from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import jensenshannon
from typing import List

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


def inference(parameters):
    val_loader = parameters['val_loader']
    model = parameters['model']
    fake_run = parameters['fake_run']
    results_basepath = parameters['results_basepath']
    gpu = parameters['gpu']
    batch_size = parameters['batch_size']
    tmp_inference_dic = dict()
    tmp_total_dic = dict()

    predicted_label_top1_list = list()
    predicted_label_top5_list = list()
    targets_list = list()
    if fake_run:
        origin_quality = dict(
            top1_acc = dict(),
            top5_acc = dict(),
        )
    loader = tqdm(val_loader, ascii=True)

    a = time.perf_counter()
    for batch_id, (images, targets) in enumerate(loader, start=1):
        if gpu is not None and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

        inference_start = time.perf_counter()
        preprocess_time = inference_start - a
        outputs = model(images)
        inference_end = time.perf_counter()
        inference_time = inference_end - inference_start
        tmp_inference_dic[batch_id] = float(inference_time)

        postprocess_start = time.perf_counter()
        for output in outputs:
            predicted_label_top1_list.append(torch.argmax(output))
            predicted_label_top5_list.append(torch.topk(output, k=5).indices)
        postprocess_end = time.perf_counter()
        postprocess_time = postprocess_end - postprocess_start
        total_time = preprocess_time + inference_time + postprocess_time
        tmp_total_dic[batch_id] = float(total_time)

        if fake_run:
            for target in targets:
                targets_list.append(target)
            batch_acc1, batch_acc5 = accuracy(predicted_label_top5_list[-batch_size:], targets[-batch_size:])
            origin_quality['top1_acc'][batch_id] = batch_acc1 
            origin_quality['top5_acc'][batch_id] = batch_acc5 

        a = time.perf_counter()
   
    if fake_run:
        acc1, acc5 = accuracy(predicted_label_top5_list, targets_list)
        print('top1-acc and top5-acc : ',acc1, acc5) # acc in the whole val set
        with open(results_basepath.joinpath('Origin_Quality.json'), 'w') as f:
            json.dump(origin_quality, f, indent=2)
                                 
    return tmp_inference_dic, tmp_total_dic


def accuracy(predicted_label_top5_list, targets):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        top1_acc = 0
        top5_acc =0
        for i, (predicted_label_top5, target) in enumerate(zip(predicted_label_top5_list, targets)):
            if predicted_label_top5[0] == target:
                top1_acc += 1
            if target in predicted_label_top5:
                top5_acc += 1
        top1_acc = "{:6.2f}".format(100*(top1_acc/len(targets)))
        top5_acc = "{:6.2f}".format(100*(top5_acc/len(targets)))
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


if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description="Run Multiple times on the Whole Test Dataset")
    parser.add_argument('--dataset-path', metavar='DIR', nargs='?', default='imagenet',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='mobilenet_v2',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: mobilenet_v2)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        metavar='N',
                        help='mini-batch size (default: 100), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    parser.add_argument('--min-run', type=int, required=False)
    parser.add_argument('--results-basepath', type=str, required=True)
    parser.add_argument('--warm-run', type=int, default=1)
    parser.add_argument('--fake-run', type=bool, default=True) # To avoid the outliers processed during the first inference
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--fit-run-number', type=int, default=2)
    parser.add_argument('--rJSD-threshold', type=float, default=0.05)
    parser.add_argument('--max-run', type=int, default=100000)
    args = parser.parse_args()

    results_basepath = Path(args.results_basepath)
    min_run = args.min_run 
    warm_run = args.warm_run 
    window_size = args.window_size
    fit_run_number = args.fit_run_number
    rJSD_threshold = args.rJSD_threshold
    fake_run = args.fake_run
    max_run = args.max_run

    dataset_root = Path(args.dataset_path)
    assert dataset_root.is_dir(), f"provided Imagenet path {dataset_root.absolute()} does not exist"

    result_path = results_basepath.joinpath('All_Times.pickle')
    rjsds_path = results_basepath.joinpath('All_rJSDs.pickle')
    fit_distribution_dir = results_basepath.joinpath('All_PDFs')
    if not fit_distribution_dir.exists():
        fit_distribution_dir.mkdir(parents=True, exist_ok=True)
    fit_distribution_model_paths = list(fit_distribution_dir.iterdir())
    fit_distribution_number = len(fit_distribution_model_paths)//2

    if result_path.exists():
        with open (result_path, 'rb') as f:
            results = pickle.load(f)
        already_run = len(results['inference'])
        del results
    else:
        already_run = 0

    # [!BEGIN] initialization of model and dataset
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    logger = set_logger(name='Tail-Quality', mode='both', level='INFO', logging_filepath=results_basepath.joinpath('Tail-Quality.log'))
    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))
    
    logger.info("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        logger.info('using CPU, this will be slow')
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    model.eval()

    valdir = os.path.join(dataset_root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    # [!END] initialization of model and dataset

    sucess_flag = False
    loop = 0 # for debugging
    with torch.no_grad():
        while not sucess_flag:
            loop += 1 # for debugging
            params = {
                'val_loader': val_loader,
                'model': model,
                'fake_run': fake_run,
                'results_basepath': results_basepath,
                'gpu': args.gpu,
                'batch_size': args.batch_size,
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
    