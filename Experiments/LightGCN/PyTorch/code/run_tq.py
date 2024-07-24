import sys
import time
import json
import numpy
import pickle
import pathlib
import logging
import argparse
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import jensenshannon
from typing import List

# for LightGCN
import world
import utils
import time
import torch
import numpy as np
import model
import register
from register import dataset


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
        bandwidth_grid = [0.005, 0.01, 0.03, 0.07, 0.1]
        best_bandwidth  = min(bandwidth_grid, key=lambda x: kde_aic(x, ins_times))
        distribution_model = KernelDensity(bandwidth=best_bandwidth).fit(ins_times)
    if fit_type == 'gmm':
        n_components_grid = [2, 3, 4, 5, 6]
        best_n_components = min(n_components_grid, key=lambda x: gmm_aic(x, ins_times))
        distribution_model = GaussianMixture(n_components=best_n_components).fit(ins_times)
    return distribution_model


def check_fit_train_test(train_times, test_times, min_ns):
    tic = time.perf_counter()
    total_js_dis = 0
    for index, (train_time, test_time, min_n) in enumerate(zip(train_times, test_times, min_ns)):
        ins_times = numpy.concatenate([train_time[:min_n], test_time])
        train_ins_times = train_time[:min_n].reshape(-1, 1)
        test_ins_times = test_time.reshape(-1, 1)
        #logger.info(f"No.{index+1} Fitting.")
        train_model = fit(train_ins_times, 'kde')
        test_model = fit(test_ins_times, 'kde')
        epsilon = 1e-8
        x = numpy.linspace(ins_times.min(), ins_times.max(), 1000).reshape(-1, 1)
        js_dis = jensenshannon(numpy.exp(train_model.score_samples(x))+epsilon, numpy.exp(test_model.score_samples(x))+epsilon)
        #js_dis = pow(jensenshannon(numpy.exp(train_model.score_samples(x))+epsilon, numpy.exp(test_model.score_samples(x))+epsilon), 2)
        #logger.info(f"No.{index} JSD: {js_dis:.3f}")
        total_js_dis += js_dis

    toc = time.perf_counter()
    logger.info(f"Total time consume: {toc-tic:.2f}s")
    avg_jsd = total_js_dis/len(min_ns)
    logger.info(f"Avg JSD: {avg_jsd:.3f}")
    return numpy.sqrt(avg_jsd)


def check_fit_dynamic(fit_distribution_models, fit_distribution_model, all_times, window_size):
    tic = time.perf_counter()
    total_js_dis = 0
    all_times = numpy.array(all_times)
    current_distribution = fit_distribution_model 
    compared_distributions = fit_distribution_models
    for compared_distribution in compared_distributions:
        epsilon = 1e-8
        x = numpy.linspace(all_times.min(), all_times.max(), 1000).reshape(-1, 1) 
        js_dis = jensenshannon(numpy.exp(current_distribution.score_samples(x))+epsilon, numpy.exp(compared_distribution.score_samples(x))+epsilon)
        total_js_dis += js_dis
    toc = time.perf_counter()
    logger.info(f"Total time consume: {toc-tic:.2f}s")
    avg_jsd = total_js_dis/window_size
    logger.info(f"Avg JSD: {avg_jsd:.3f}")
    logger.info(f"rJSD: {numpy.sqrt(avg_jsd):.3f}")
    return numpy.sqrt(avg_jsd)


def test_one_batch(X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r = utils.getLabel(groundTrue, sorted_items)
        pre, recall, ndcg = [], [], []
        for k in world.topks:
            ret = utils.RecallPrecision_ATk(groundTrue, r, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
        return {'recall':np.array(recall), 
                'precision':np.array(pre), 
                'ndcg':np.array(ndcg)}


def Inference(params):
    dataset = params['dataset']
    Recmodel = params['Recmodel']
    fake_run = params['fake_run']
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout

    max_K = max(world.topks)
    
    users = list(testDict.keys())
    try:
        assert u_batch_size <= len(users) / 10
    except AssertionError:
        logger.info(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
    users_list = []
    rating_list = []
    groundTrue_list = []
        
    total_batch = len(users) // u_batch_size + 1
    tmp_inference_dic = dict()
    tmp_total_dic = dict()
    a = time.perf_counter()
    for batch_id, batch_users in enumerate(utils.minibatch(users, batch_size=u_batch_size), start=1):
        allPos = dataset.getUserPosItems(batch_users)
        groundTrue = [testDict[u] for u in batch_users]
        batch_users_gpu = torch.Tensor(batch_users).long()
        batch_users_gpu = batch_users_gpu.to(world.device)

        inference_start = time.perf_counter()
        preprocess_time = inference_start - a 
        rating = Recmodel.getUsersRating(batch_users_gpu)
        inference_end = time.perf_counter()
        inference_time = inference_end - inference_start
        tmp_inference_dic[batch_id] = float(inference_time)
        postprocess_start = time.perf_counter()
        
        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)
        rating[exclude_index, exclude_items] = -(1<<10)
        _, rating_K = torch.topk(rating, k=max_K)
        rating = rating.cpu().numpy()
        
        del rating
        users_list.append(batch_users)
        rating_list.append(rating_K.cpu())
        groundTrue_list.append(groundTrue)
        postprocess_end = time.perf_counter()
        postprocess_time = postprocess_end - postprocess_start
        total_time = preprocess_time + inference_time + postprocess_time
        tmp_total_dic[batch_id] = float(total_time)
        # logger.info('total_time, inference_time: ', total_time, inference_time)
        a = time.perf_counter()

    assert total_batch == len(users_list)

    X = zip(rating_list, groundTrue_list)

    pre_results = []
    for batch_id, x in enumerate(X, start=1):
        pre_results.append(test_one_batch(x))
        
    if fake_run:
        origin_quality = dict()   
        for batch_id, result in enumerate(pre_results, start=1):
            origin_quality[batch_id] = dict(
                recall = float(result['recall'].item()),
                precision = float(result['precision'].item()),
                ndcg = float(result['ndcg'].item())
            )
        with open(results_basepath.joinpath('Light_GCN_Pytorch_origin_quality.json'), 'w') as f:
            json.dump(origin_quality, f, indent=2)
            
    return  tmp_inference_dic, tmp_total_dic
    
    # for result in pre_results: 
    #     results['recall'] += result['recall']
    #     results['precision'] += result['precision']
    #     results['ndcg'] += result['ndcg']
    # results['recall'] /= float(len(users))
    # results['precision'] /= float(len(users))
    # results['ndcg'] /= float(len(users))


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
    plt.savefig(results_basepath.joinpath("Light_GCN_Pytorch_rjsds.jpg"), format="jpg")
    plt.savefig(results_basepath.joinpath("Light_GCN_Pytorch_rjsds.pdf"), format="pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multiple times on the Whole Test Dataset")
    parser.add_argument('--min-run', type=int, required=False)
    parser.add_argument('--batch-size', type=int, default=1) # for LightGCN, using the offical method to set batch-size
    parser.add_argument('--dataset-path', type=str, default="") # for LightGCN, using the offical method to load dataset
    parser.add_argument('--results-basepath', type=str, required=True)
    parser.add_argument('--model-path', type=str, default="") # for LightGCN, using the offical method to load checkpoint
    parser.add_argument('--warm-run', type=int, default=1)
    parser.add_argument('--fake-run', type=bool, default=True) # To avoid the outliers processed during the first inference
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--fit-run-number', type=int, default=2)
    parser.add_argument('--rJSD-threshold', type=float, default=0.05)
    parser.add_argument('--max-run', type=int, default=0)
    args = parser.parse_args()


    results_basepath = pathlib.Path(args.results_basepath)
    min_run = args.min_run 
    warm_run = args.warm_run 
    window_size = args.window_size
    fit_run_number = args.fit_run_number
    rJSD_threshold = args.rJSD_threshold
    fake_run = args.fake_run
    max_run = args.max_run

    result_path = results_basepath.joinpath('Light_GCN_Pytorch.pickle')
    rjsds_path = results_basepath.joinpath('Light_GCN_Pytorch_rjsds.pickle')
    fit_distribution_path = results_basepath.joinpath('Light_GCN_Pytorch_distributions.pickle')
    logger = set_logger(name='Light_GCN_Pytorch', mode='both', level='INFO', logging_filepath=results_basepath.joinpath('Light_GCN_Pytorch.log'))

    total_batches = 0
    if result_path.exists():
        with open (result_path, 'rb') as f:
            results = pickle.load(f)
        already_run = len(results)
        del results
    else:
        already_run = 0

    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    weight_file = utils.getFileName()
    logger.info(f"load and save to {weight_file}")
    if world.LOAD:
        try:
            Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cuda')))
            logger.info(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            logger.info(f"{weight_file} not exists, start from beginning")
    sucess_flag = False
    Recmodel.eval()
    loop = 0 # for debugging

    with torch.no_grad():
        while not sucess_flag:
            loop += 1 # for debugging
            params = {
                'dataset': dataset,
                'Recmodel': Recmodel,
                'fake_run': fake_run
            }

            logger.info(f'-------before inference {loop}-------')
            logger.info(f'already_run: {already_run}')
            logger.info(f'warm_run: {warm_run}')
            
            tmp_inference_dic, tmp_total_dic = Inference(params)
            if not fake_run:
                already_run += 1   
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
                with open(result_path, 'wb') as f:
                    pickle.dump(tmp_results, f)

                for key, value in tmp_inference_dic.items():
                    all_inference_times.append(value)
                for key, value in tmp_total_dic.items():
                    all_total_times.append(value)
                if (already_run - warm_run) % fit_run_number == 0:
                    fit_inference_distribution_model = fit(all_inference_times) 
                    fit_total_distribution_model = fit(all_total_times) 

                if fit_distribution_path.exists():
                    with open(fit_distribution_path, 'rb') as f:
                        fit_distribution_models = pickle.load(f)
                        tmp_fit_distribution_models = fit_distribution_models.copy()
                        del fit_distribution_models 
                        
                else:
                    tmp_fit_distribution_models = dict(
                        inference = list(),
                        total = list()
                    )
                    
                logger.info(f"len(tmp_fit_distribution_models['inference']) % window_size == {len(tmp_fit_distribution_models['inference']) % window_size}")
                if len(tmp_fit_distribution_models['inference']) % window_size == 0 and len(tmp_fit_distribution_models['inference']) != 0:
                    inference_rjsd = check_fit_dynamic(tmp_fit_distribution_models['inference'][-window_size:], fit_inference_distribution_model, all_inference_times, window_size)
                    total_rjsd = check_fit_dynamic(tmp_fit_distribution_models['total'][-window_size:], fit_total_distribution_model, all_total_times, window_size)
                    logger.info(f'inference_rjsd is {inference_rjsd} / total_rjsd is {total_rjsd}')
                    sucess_flag = True if inference_rjsd <= rJSD_threshold and total_rjsd <= rJSD_threshold else False
                    if inference_rjsd <= rJSD_threshold:
                        logger.info('inference_times has fitted') 
                    if total_rjsd <= rJSD_threshold:
                        logger.info('total_times has fitted') 
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

                tmp_fit_distribution_models['inference'].append(fit_inference_distribution_model)
                tmp_fit_distribution_models['total'].append(fit_total_distribution_model)
                with open(fit_distribution_path, 'wb') as f:
                    pickle.dump(tmp_fit_distribution_models, f)

                del tmp_results
                del tmp_fit_distribution_models
                del all_total_times
                del all_inference_times

            
            logger.info(f'-------after inference {loop}-------')
            logger.info(f'already_run: {already_run}')
            logger.info(f'warm_run: {warm_run}')
            if fake_run:
                logger.info(f'this run is fake')

            fake_run = False

            if already_run == max_run:
                break 
            

            
    

