import time
import json
import numpy
import pickle
import pathlib
import argparse
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import jensenshannon
from typing import List

import world
import utils
import time
import torch
import numpy as np
import model
import register
from register import dataset


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
        #print(f"No.{index+1} Fitting.")
        train_model = fit(train_ins_times, 'kde')
        test_model = fit(test_ins_times, 'kde')
        epsilon = 1e-8
        x = numpy.linspace(ins_times.min(), ins_times.max(), 1000).reshape(-1, 1)
        js_dis = jensenshannon(numpy.exp(train_model.score_samples(x))+epsilon, numpy.exp(test_model.score_samples(x))+epsilon)
        #js_dis = pow(jensenshannon(numpy.exp(train_model.score_samples(x))+epsilon, numpy.exp(test_model.score_samples(x))+epsilon), 2)
        #print(f"No.{index} JSD: {js_dis:.3f}")
        total_js_dis += js_dis

    toc = time.perf_counter()
    print(f"Total time consume: {toc-tic:.2f}s")
    avg_jsd = total_js_dis/len(min_ns)
    print(f"Avg JSD: {avg_jsd:.3f}")
    return numpy.sqrt(avg_jsd)


def check_fit_dynamic(fit_distribution_models, fit_distribution_model, all_times, window_size):
    tic = time.perf_counter()
    total_js_dis = 0
    all_times = numpy.array(all_times)
    current_distribution = fit_distribution_model 
    compared_distributions = fit_distribution_models[-window_size:]
    for compared_distribution in compared_distributions:
        epsilon = 1e-8
        x = numpy.linspace(all_times.min(), all_times.max(), 1000).reshape(-1, 1) 
        js_dis = jensenshannon(numpy.exp(current_distribution.score_samples(x))+epsilon, numpy.exp(compared_distribution.score_samples(x))+epsilon)
        total_js_dis += js_dis
    toc = time.perf_counter()
    print(f"Total time consume: {toc-tic:.2f}s")
    avg_jsd = total_js_dis/window_size
    print(f"Avg JSD: {avg_jsd:.3f}")
    print(f"rJSD: {numpy.sqrt(avg_jsd):.3f}")
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
    warm_run = params['warm_run']
    already_run_number = params['already_run_number']
    already_batch_number = params['already_batch_number']
    warm_batch_number = params['warm_batch_number']
    fit_distribution_models = params['fit_distribution_models']
    rjsds = params['rjsds']
    rJSD_threshold = params['rJSD_threshold']
    sucess_flag = params['sucess_flag']
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
        print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
    users_list = []
    rating_list = []
    groundTrue_list = []
        
    total_batch = len(users) // u_batch_size + 1
    tmp_dic = dict()
    for batch_id, batch_users in enumerate(utils.minibatch(users, batch_size=u_batch_size), start=1):
        allPos = dataset.getUserPosItems(batch_users)
        groundTrue = [testDict[u] for u in batch_users]
        batch_users_gpu = torch.Tensor(batch_users).long()
        batch_users_gpu = batch_users_gpu.to(world.device)

        inference_start = time.perf_counter()
        rating = Recmodel.getUsersRating(batch_users_gpu)
        inference_end = time.perf_counter()
        inference_time = inference_end - inference_start
        tmp_dic[batch_id] = float(inference_time)
        all_times.append(inference_time)

        already_batch_number += 1
        if already_run_number < warm_run: 
            warm_batch_number += 1 
        if already_run_number is not 0 and ((already_batch_number - warm_batch_number) % fit_batches) == 0:
            fit_distribution_model = fit(all_times)
            print(' len(fit_distribution_models) % window_size : ', len(fit_distribution_models) % window_size) 
            if len(fit_distribution_models) % window_size == 0 and len(fit_distribution_models) is not 0:
                rjsd = check_fit_dynamic(fit_distribution_models, fit_distribution_model, all_times, window_size)
                rjsds.append(rjsd)
                print('During this run, rjsd is', rjsd)
                sucess_flag = True if rjsd <= rJSD_threshold else False

            fit_distribution_models.append(fit_distribution_model)
            
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
    assert total_batch == len(users_list)
    
    X = zip(rating_list, groundTrue_list)

    pre_results = []
    for batch_id, x in enumerate(X, start=1):
        post_process_start = time.perf_counter()
        pre_results.append(test_one_batch(x))
        post_process_end = time.perf_counter()

    if already_run_number == 0:
        origin_quality = dict()   
        for batch_id, result in enumerate(pre_results, start=1):
            origin_quality[batch_id] = dict(
                recall = float(result['recall'].item()),
                precision = float(result['precision'].item()),
                ndcg = float(result['ndcg'].item())
            )
        
        with open(results_basepath.joinpath('Light_GCN_Pytorch_origin_quality.json'), 'w') as f:
            json.dump(origin_quality, f, indent=2)
    already_run_number += 1
    if fake_run:
        return tmp_dic, params['already_run_number'], params['already_batch_number'], params['warm_batch_number'], params['fit_distribution_models'], params['rjsds'], params['sucess_flag']
    else:
        return tmp_dic, already_run_number, already_batch_number, warm_batch_number, fit_distribution_models, rjsds, sucess_flag
    # for result in pre_results:
    #     results['recall'] += result['recall']
    #     results['precision'] += result['precision']
    #     results['ndcg'] += result['ndcg']
        
    # results['recall'] /= float(len(users))
    # results['precision'] /= float(len(users))
    # results['ndcg'] /= float(len(users))


def draw_rjsds(rjsds: List, results_basepath: pathlib.Path):
    x_data = list(range(1, len(rjsds) + 1))
    fig, ax = plt.subplots()
    ax.plot(x_data, rjsds, marker='o', linestyle='-', color='b', label='rJSD')
    ax.set_title('rJSD Fitting Progress')
    ax.set_xlabel('Fitting Round')
    ax.set_ylabel('rJSD')
    ax.grid(True)
    plt.savefig(results_basepath.joinpath("Light_GCN_Pytorch_rjsds.jpg"), format="jpg")
    plt.savefig(results_basepath.joinpath("Light_GCN_Pytorch_rjsds.pdf"), format="pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multiple times on the Whole Test Dataset")
    parser.add_argument('--min-run-number', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=1) # for LightGCN, using the offical method to set batch-size
    parser.add_argument('--dataset-path', type=str, default="") # for LightGCN, using the offical method to load dataset
    parser.add_argument('--results-basepath', type=str, required=True)
    parser.add_argument('--model-path', type=str, default="") # for LightGCN, using the offical method to load checkpoint
    parser.add_argument('--warm-run', type=int, default=1)
    parser.add_argument('--fake-run', type=bool, default=True) # To avoid the outliers processed during the first inference
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--fit-batches', type=int, default=100)
    parser.add_argument('--rJSD-threshold', type=float, default=0.05)
    parser.add_argument('--test-run', type=int, default=0)
    args = parser.parse_args()

    # dataset_path = pathlib.Path(args.dataset_path)
    # model_path = pathlib.Path(args.model_path)
    results_basepath = pathlib.Path(args.results_basepath)
    min_run_number = args.min_run_number 
    warm_run = args.warm_run 
    window_size = args.window_size
    fit_batches = args.fit_batches
    rJSD_threshold = args.rJSD_threshold
    fake_run = args.fake_run
    test_run = args.test_run

    result_path = results_basepath.joinpath('Light_GCN_Pytorch.json')
    rjsds_path = results_basepath.joinpath('Light_GCN_Pytorch_rjsds.json')
    fit_distribution_path = results_basepath.joinpath('Light_GCN_Pytorch_distributions.pickle')

    all_times = list()
    total_batches = 0
    if result_path.exists():
        with open (result_path, 'r') as f:
            results = json.load(f)
            for run in results:
                total_batches += len(run.keys())
                for inference_time in run.values():
                    all_times.append(inference_time)
        with open(rjsds_path, 'r') as f:
            rjsds = json.load(f)
        with open(fit_distribution_path, 'rb') as f:
            fit_distribution_models = pickle.load(f)
    
        already_run_number = len(results)
        already_batch_number = total_batches 
        warm_batch_number = len(results[0].keys())
    else:
        results = list()
        fit_distribution_models = list()
        rjsds = list()
        already_run_number = 0
        already_batch_number = 0
        warm_batch_number = 0

    """
        results = [
            { 
                "batch_1" : float,
                ...
            }
            {
                "batch_2" : float,
                ...
            }
            ...
        ]
    """

    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    Recmodel = Recmodel.to(world.device)

    weight_file = utils.getFileName()
    print(f"load and save to {weight_file}")
    if world.LOAD:
        try:
            Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cuda')))
            print(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    sucess_flag = False
    Recmodel.eval()
    loop = 0 # for debugging

    with torch.no_grad():
        while not sucess_flag:
            loop += 1 # for debugging
            params = {
                'dataset': dataset,
                'Recmodel': Recmodel,
                'fake_run': fake_run,
                'warm_run': warm_run,
                'already_run_number': already_run_number,
                'already_batch_number': already_batch_number,
                'warm_batch_number': warm_batch_number,
                'fit_distribution_models': fit_distribution_models,
                'rjsds': rjsds,
                'rJSD_threshold': rJSD_threshold,
                'sucess_flag': sucess_flag
            }

            print('-------before_inference-------')
            print(f'in loop {loop}:', loop)
            print(f'in run {already_run_number} :')
            print('already_batch_number: ', already_batch_number)
            print('warm_batch_number: ', warm_batch_number)
            print('fitted_models: ', len(fit_distribution_models))
            print('rjsds:', len(rjsds))

            result, already_run_number, already_batch_number, warm_batch_number, \
            fit_distribution_models, rjsds, sucess_flag = Inference(params)
            if not fake_run:
                results.append(result)
            fake_run = False
            
            print('-------after_inference-------')
            print(f'in loop {loop}:', loop)
            print(f'in already_run_number: {already_run_number}')
            print('already_batch_number: ', already_batch_number)
            print('warm_batch_number: ', warm_batch_number)
            print('fitted_models: ', len(fit_distribution_models))
            print('rjsds_nums:', len(rjsds))
            if already_run_number == test_run:
                break 

    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    with open(rjsds_path, 'w') as f:
        json.dump(rjsds, f, indent=2)
    with open(fit_distribution_path, 'wb') as f:
        pickle.dump(fit_distribution_models, f)
    draw_rjsds(rjsds, results_basepath) 
    

