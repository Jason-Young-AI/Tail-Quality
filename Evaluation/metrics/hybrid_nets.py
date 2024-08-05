import numpy

from . import Task
from ..utils.io import load_json, load_pickle
from ..utils.expand import expand_indexed_batches, expand_round_time


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = numpy.concatenate(([0.0], recall, [1.0]))
    mpre = numpy.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = numpy.flip(numpy.maximum.accumulate(numpy.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = numpy.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = numpy.trapz(numpy.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = numpy.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = numpy.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls, names=[]):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = numpy.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = numpy.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    px, py = numpy.linspace(0, 1, 1000), []  # for plotting
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = numpy.zeros(s), numpy.zeros((unique_classes.shape[0], 1000)), numpy.zeros((unique_classes.shape[0], 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score
            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    i=r.mean(0).argmax()

    return p[:, i], r[:, i], f1[:, i], ap, unique_classes.astype('int32')


class HybridNets(Task):
    @classmethod
    def pre_process(cls, goldens_filepath, results_filepath, alltime_filepath, alltime_type) -> tuple[None, list[str], list[list[float]]]:
        results = load_pickle(results_filepath)
        results, results_batch_sizes = expand_indexed_batches(results)

        alltime = load_pickle(alltime_filepath)[alltime_type]
        multiple_inference_times: list[list[float]] = list()
        for round_time in alltime:
            round_time = expand_round_time(round_time, results_batch_sizes)
            multiple_inference_times.append(round_time)

        return None, results, multiple_inference_times


    @classmethod
    def calculate_metrics(cls, goldens: None, results: list[int], validities: numpy.ndarray) -> tuple[float, float, float, float, float]:
        # Return: Recall, mAP@50, mIoU, Acc, IoU
        # Traffic Object Detection - (Recall, mAP@50)
        # Drivable Area Segmentation - (mIoU)
        # Lane Line Detection - (Acc, IoU)
        # each element of inference_results must be a list
        masked_stats = list()
        masked_iou_ls = list()
        masked_acc_ls = list()
        for result, validity in zip(results, validities):
            stats = result['stats']
            iou_ls = result['iou_ls']
            acc_ls = result['acc_ls']
            if not validity:
                stats = (numpy.zeros(0, 10, dtype=bool), numpy.array([]), numpy.array([]), stats[-1])
                iou_ls = [iou_ls*0 for each_iou_ls in iou_ls]
                acc_ls = [acc_ls*0 for each_acc_ls in acc_ls]
            masked_stats.append(stats)
            masked_iou_ls.append(iou_ls)
            masked_acc_ls.append(acc_ls)

        ncs = 3
        stats = masked_stats
        iou_ls = masked_iou_ls
        acc_ls = masked_acc_ls
        for i in range(ncs):
            iou_ls[i] = numpy.concatenate(iou_ls[i])
            acc_ls[i] = numpy.concatenate(acc_ls[i])
        iou_score = numpy.mean(iou_ls)
        acc_score = numpy.mean(acc_ls)

        miou_ls = []
        seg_list = ['road', 'lane']
        for i in range(len(seg_list)):
            miou_ls.append(numpy.mean( (iou_ls[0] + iou_ls[i+1]) / 2))

        for i in range(ncs):
            iou_ls[i] = numpy.mean(iou_ls[i])
            acc_ls[i] = numpy.mean(acc_ls[i])

        # Compute statistics
        stats = [numpy.concatenate(x, 0) for x in zip(*stats)]

        names = ['car']
        ap50 = None
        # Compute metrics
        if len(stats) and stats[0].any():
            p, r, f1, ap, ap_class = ap_per_class(*stats, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = numpy.bincount(stats[3].astype(numpy.int64), minlength=1)  # number of targets per class
        else:
            nt = numpy.zeros(1)

        return mr, map50, iou_score, miou_ls[0], acc_ls[2], iou_ls[2]