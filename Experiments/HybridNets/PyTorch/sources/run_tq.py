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

from tqdm import tqdm
from utils import smp_metrics
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import jensenshannon
from typing import List
from utils.utils import ConfusionMatrix, postprocess, scale_coords, process_batch, ap_per_class, fitness, \
    save_checkpoint, DataLoaderX, BBoxTransform, ClipBoxes, boolean_string, Params
from hybridnets.dataset import BddDataset
from utils.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
from torchvision import transforms
import torch.nn.functional as F
from backbone import HybridNetsBackbone
from hybridnets.model import ModelWithLoss

# for LightGCN
import time
import torch
import numpy as np


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
    val_generator = parameters['dataset']
    model = parameters['hybridnets']
    fake_run = parameters['fake_run']
    opt = parameters['opt']
    params = parameters['params']
    seg_mode = parameters['seg_mode']

    loss_regression_ls = []
    loss_classification_ls = []
    loss_segmentation_ls = []
    stats, ap, ap_class = [], [], []
    iou_thresholds = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    num_thresholds = iou_thresholds.numel()
    names = {i: v for i, v in enumerate(params.obj_list)}
    nc = len(names)
    ncs = 1 if seg_mode == BINARY_MODE else len(params.seg_list) + 1
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    s_seg = ' ' * (15 + 11 * 8)
    s = ('%-15s' + '%-11s' * 8) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'mIoU', 'mAcc')
    for i in range(len(params.seg_list)):
            s_seg += '%-33s' % params.seg_list[i]
            s += ('%-11s' * 3) % ('mIoU', 'IoU', 'Acc')
    p, r, f1, mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    iou_ls = [[] for _ in range(ncs)]
    acc_ls = [[] for _ in range(ncs)]
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    tmp_inference_dic = dict()
    tmp_total_dic = dict()
    a = time.perf_counter()

    val_loader = tqdm(val_generator, ascii=True)
    for batch_id, data in enumerate(val_loader):
        imgs = data['img']
        annot = data['annot']
        seg_annot = data['segmentation']
        filenames = data['filenames']
        shapes = data['shapes']

        if opt.num_gpus == 1:
            imgs = imgs.cuda()
            annot = annot.cuda()
            seg_annot = seg_annot.cuda()


        inference_start = time.perf_counter()
        preprocess_time = inference_start - a 
        cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation = model(imgs, annot,
                                                                                                seg_annot,
                                                                                                obj_list=params.obj_list)
        inference_end = time.perf_counter()
        inference_time = inference_end - inference_start
        tmp_inference_dic[batch_id] = float(inference_time)
        postprocess_start = time.perf_counter()

        out = postprocess(imgs.detach(),
                            torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regression.detach(),
                            classification.detach(),
                            regressBoxes, clipBoxes,
                            opt.conf_thres, opt.iou_thres)  # 0.5, 0.3

        postprocess_end = time.perf_counter()
        postprocess_time = postprocess_end - postprocess_start
        total_time = preprocess_time + inference_time + postprocess_time
        # print(inference_time)
        # print(total_time)
        tmp_total_dic[batch_id] = float(total_time)

        if fake_run:
            for i in range(annot.size(0)):
                seen += 1
                labels = annot[i]
                labels = labels[labels[:, 4] != -1]

                ou = out[i]
                nl = len(labels)

                pred = np.column_stack([ou['rois'], ou['scores']])
                pred = np.column_stack([pred, ou['class_ids']])
                pred = torch.from_numpy(pred).cuda()

                target_class = labels[:, 4].tolist() if nl else []  # target class

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, num_thresholds, dtype=torch.bool),
                                    torch.Tensor(), torch.Tensor(), target_class))
                    # print("here")
                    continue

                if nl:
                    pred[:, :4] = scale_coords(imgs[i][1:], pred[:, :4], shapes[i][0], shapes[i][1])
                    labels = scale_coords(imgs[i][1:], labels, shapes[i][0], shapes[i][1])
                    # ori_img = cv2.imread('datasets/bdd100k_effdet/val/' + filenames[i],
                    #                      cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_UNCHANGED)
                    # for label in labels:
                    #     x1, y1, x2, y2 = [int(x) for x in label[:4]]
                    #     ori_img = cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    # for pre in pred:
                    #     x1, y1, x2, y2 = [int(x) for x in pre[:4]]
                    #     # ori_img = cv2.putText(ori_img, str(pre[4].cpu().numpy()), (x1 - 10, y1 - 10),
                    #     #                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    #     ori_img = cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

                    # cv2.imwrite('pre+label-{}.jpg'.format(filenames[i]), ori_img)
                    correct = process_batch(pred, labels, iou_thresholds)
                    if opt.plots:
                        confusion_matrix.process_batch(pred, labels)
                else:
                    correct = torch.zeros(pred.shape[0], num_thresholds, dtype=torch.bool)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_class))

                # print(stats)

                # Visualization
                # seg_0 = segmentation[i]
                # # print('bbb', seg_0.shape)
                # seg_0 = torch.argmax(seg_0, dim = 0)
                # # print('before', seg_0.shape)
                # seg_0 = seg_0.cpu().numpy()
                #     #.transpose(1, 2, 0)
                # # print(seg_0.shape)
                # anh = np.zeros((384,640,3))
                # anh[seg_0 == 0] = (255,0,0)
                # anh[seg_0 == 1] = (0,255,0)
                # anh[seg_0 == 2] = (0,0,255)
                # anh = np.uint8(anh)
                # cv2.imwrite('segmentation-{}.jpg'.format(filenames[i]),anh)         
            if seg_mode == MULTICLASS_MODE:
                segmentation = segmentation.log_softmax(dim=1).exp()
                _, segmentation = torch.max(segmentation, 1)  # (bs, C, H, W) -> (bs, H, W)
            else:
                segmentation = F.logsigmoid(segmentation).exp()

            tp_seg, fp_seg, fn_seg, tn_seg = smp_metrics.get_stats(segmentation, seg_annot, mode=seg_mode,
                                                                threshold=0.5 if seg_mode != MULTICLASS_MODE else None,
                                                                num_classes=ncs if seg_mode == MULTICLASS_MODE else None)
            iou = smp_metrics.iou_score(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')
            #         print(iou)
            acc = smp_metrics.balanced_accuracy(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')

            for i in range(ncs):
                iou_ls[i].append(iou.T[i].detach().cpu().numpy())
                acc_ls[i].append(acc.T[i].detach().cpu().numpy())

        if fake_run:
            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()
            seg_loss = seg_loss.mean()
            loss = cls_loss + reg_loss + seg_loss
            if loss == 0 or not torch.isfinite(loss):
                continue

            loss_classification_ls.append(cls_loss.item())
            loss_regression_ls.append(reg_loss.item())
            loss_segmentation_ls.append(seg_loss.item())

        # logger.info('total_time, inference_time: ', total_time, inference_time)
        a = time.perf_counter()

    if fake_run:
        cls_loss = np.mean(loss_classification_ls)
        reg_loss = np.mean(loss_regression_ls)
        seg_loss = np.mean(loss_segmentation_ls)
        loss = cls_loss + reg_loss + seg_loss

        print(
            'Val. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Segmentation loss: {:1.5f}. Total loss: {:1.5f}'.format(
                cls_loss, reg_loss, seg_loss, loss))

        for i in range(ncs):
            iou_ls[i] = np.concatenate(iou_ls[i])
            acc_ls[i] = np.concatenate(acc_ls[i])
        # print(len(iou_ls[0]))
        iou_score = np.mean(iou_ls)
        # print(iou_score)
        acc_score = np.mean(acc_ls)

        miou_ls = []
        for i in range(len(params.seg_list)):
            if seg_mode == BINARY_MODE:
                # typically this runs once with i == 0
                miou_ls.append(np.mean(iou_ls[i]))
            else:
                miou_ls.append(np.mean( (iou_ls[0] + iou_ls[i+1]) / 2))

        for i in range(ncs):
            iou_ls[i] = np.mean(iou_ls[i])
            acc_ls[i] = np.mean(acc_ls[i])

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        # print(stats[3])

        # Count detected boxes per class
        # boxes_per_class = np.bincount(stats[2].astype(np.int64), minlength=1)

        ap50 = None
        save_dir = 'plots'
        os.makedirs(save_dir, exist_ok=True)

        # Compute metrics
        if len(stats) and stats[0].any():
            p, r, f1, ap, ap_class = ap_per_class(*stats, plot=opt.plots, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=1)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        print(s_seg)
        print(s)
        pf = ('%-15s' + '%-11i' * 2 + '%-11.3g' * 6) % ('all', seen, nt.sum(), mp, mr, map50, map, iou_score, acc_score)
        for i in range(len(params.seg_list)):
            tmp = i+1 if seg_mode != BINARY_MODE else i
            pf += ('%-11.3g' * 3) % (miou_ls[i], iou_ls[tmp], acc_ls[tmp])
        print(pf)

        # Print results per class
        if opt.verbose and nc > 1 and len(stats):
            pf = '%-15s' + '%-11i' * 2 + '%-11.3g' * 4
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Plots
        if opt.plots:
            confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
            confusion_matrix.tp_fp()

        results = (mp, mr, map50, map, iou_score, acc_score, loss)

        with open(results_basepath.joinpath('HybridNets_Pytorch_origin_quality.json'), 'w') as f:
            json.dump(results, f, indent=2)

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
    plt.savefig(results_basepath.joinpath("Light_GCN_Pytorch_rjsds.jpg"), format="jpg")
    plt.savefig(results_basepath.joinpath("Light_GCN_Pytorch_rjsds.pdf"), format="pdf")


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

    parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
    parser.add_argument('-bb', '--backbone', type=str,
                   help='Use timm to create another backbone replacing efficientnet. '
                   'https://github.com/rwightman/pytorch-image-models')
    parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficients of efficientnet backbone')
    parser.add_argument('-w', '--weights', type=str, default='weights/hybridnets.pth', help='/path/to/weights')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='Num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('-v', '--verbose', type=boolean_string, default=True,
                    help='Whether to print results per class when valing')
    parser.add_argument('--cal_map', type=boolean_string, default=True,
                        help='Calculate mAP in validation')
    parser.add_argument('--plots', type=boolean_string, default=True,
                    help='Whether to plot confusion matrix when valing')
    parser.add_argument('--num_gpus', type=int, default=1,
                    help='Number of GPUs to be used (0 to use CPU)')
    parser.add_argument('--conf_thres', type=float, default=0.001,
                    help='Confidence threshold in NMS')
    parser.add_argument('--iou_thres', type=float, default=0.6,
                    help='IoU threshold in NMS')
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
    fit_distribution_dir = results_basepath.joinpath('distributions')
    if not fit_distribution_dir.exists():
        fit_distribution_dir.mkdir(parents=True, exist_ok=True)
    fit_distribution_model_paths = list(fit_distribution_dir.iterdir())
    fit_distribution_number = len(fit_distribution_model_paths)//2

    logger = set_logger(name='Light_GCN_Pytorch', mode='both', level='INFO', logging_filepath=results_basepath.joinpath('Light_GCN_Pytorch.log'))
    total_batches = 0
    if result_path.exists():
        with open (result_path, 'rb') as f:
            results = pickle.load(f)
        already_run = len(results['inference'])
        del results
    else:
        already_run = 0

    # [!Begin] Model Initialization

    compound_coef = args.compound_coef
    project_name = args.project
    weights_path = args.model_path

    params = Params(f'projects/{project_name}.yml')
    obj_list = params.obj_list
    seg_mode = MULTILABEL_MODE if params.seg_multilabel else MULTICLASS_MODE if len(params.seg_list) > 1 else BINARY_MODE

    valid_dataset = BddDataset(
        params=params,
        is_train=False,
        inputsize=params.model['image_size'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=params.mean, std=params.std
            )
        ]),
        seg_mode=seg_mode
    )

    val_generator = DataLoaderX(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=params.pin_memory,
        collate_fn=BddDataset.collate_fn
    )
    model = HybridNetsBackbone(compound_coef=compound_coef, num_classes=len(params.obj_list),
                               ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales),
                               seg_classes=len(params.seg_list), backbone_name=args.backbone,
                               seg_mode=seg_mode)
    
    try:
        model.load_state_dict(torch.load(weights_path))
    except:
        model.load_state_dict(torch.load(weights_path)['model'])
    model = ModelWithLoss(model, debug=False)
    model.requires_grad_(False)

    if args.num_gpus > 0:
        model.cuda()

    model.eval()

    # [!End] Model Initialization

    sucess_flag = False
    loop = 0 # for debugging
    with torch.no_grad():
        while not sucess_flag:
            loop += 1 # for debugging
            params = {
                'dataset': val_generator,
                'hybridnets': model,
                'fake_run': fake_run,
                'opt': args,
                'params': params,
                'seg_mode': seg_mode
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
            

            
    

