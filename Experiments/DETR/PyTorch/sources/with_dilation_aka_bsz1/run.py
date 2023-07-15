import os
import time
import json
import base64
import requests
import argparse

import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from transforms import Compose, DetermineResize, ToTensor, Normalize
from pycocotools.coco import COCO
from box_ops import box_cxcywh_to_xyxy
from utils import prepare_for_coco_detection
from misc import nested_tensor_from_tensor_list


image_preprocessing_local = Compose([
    DetermineResize(800, max_size=1333),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


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


def add_other(coco_results, image_sizes, inference_time, pre_time, post_time, image_shape):
    coco_results_with_other = list()
    for coco_result, image_size in zip(coco_results, image_sizes):
        coco_results_with_other.append(
            dict(
                origin_image_size=image_size,
                batch_image_size=image_shape,
                result=coco_result,
                inference_time=inference_time,
                postprocess_time=post_time,
                preprocess_time=pre_time,
            )
        )

    return coco_results_with_other


def load_dataset(anno_path, image_dir):
    coco = COCO(anno_path)
    image_ids = list(sorted(coco.imgs.keys()))
    img_paths = list()

    for image_id in image_ids:
        image_name = coco.loadImgs(image_id)[0]["file_name"]
        img_paths.append(os.path.join(image_dir, image_name))

    return img_paths, image_ids


def req_post(dest, batch, model=None):
    if model is None:
        images = dict()
        for index, (img_path, img_id) in enumerate(batch):
            img = open(img_path, 'rb').read()
            img_str = base64.b64encode(img).decode('utf-8')
            images[index] = (img_str, img_id)

        headers = {"Content-type": "application/json", "Accept": "text/plain"}
        images = json.dumps(images)
        response = requests.post(dest, data=images, headers=headers)
        results, image_sizes, inference_time, pre_time, post_time, batch_image_shape = response.json()
    else:
        start_time = time.perf_counter()
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

        images = nested_tensor_from_tensor_list(images).to('cuda:0')
        image_sizes = torch.stack(image_sizes, dim=0).to('cuda:0')
        stop_time = time.perf_counter()
        pre_time = stop_time - start_time

        start_time = time.perf_counter()
        results = model(images)
        stop_time = time.perf_counter()
        inference_time = stop_time - start_time

        start_time = time.perf_counter()
        results = postprocess(results, image_sizes, image_indices)
        stop_time = time.perf_counter()
        post_time = stop_time - start_time
        image_sizes = image_sizes.tolist()
        batch_image_shape = images.tensors.shape[-2:]

    results = add_other(results, image_sizes, inference_time, pre_time, post_time, batch_image_shape)
    return results
    #res = requests.post(dest, files={'data': open('/home/zxyang/1.Datasets/coco2017/val2017/000000581781.jpg', 'rb'), 'data': open('/home/zxyang/1.Datasets/coco2017/val2017/000000581482.jpg', 'rb')})


def run_all(start_i, run_number, batch_size, server_url, img_paths, img_ids, results_basepath, write_main_indices=[1,], warm_run=1, model=None):
    assert isinstance(write_main_indices, set), "Wrong type of argument \"write_main_indices\": \"{write_main_indices}\""
    assert len(img_paths) == len(img_ids), "Fatal Error!"
    print(f" + Total warmup round - {warm_run}")
    print(f" + After that process will start from round {start_i} to {run_number}")
    for true_i, i in enumerate(range(start_i, run_number + warm_run + 1)):
        results = list()
        batch = list()
        num = 0
        for index, (img_path, img_id) in enumerate(zip(img_paths, img_ids)):
            batch.append((img_path, img_id))
            if len(batch) == batch_size or index == len(img_paths) - 1:
                num += 1
                response = req_post(server_url, batch, model)
                results.append(response)
                print(f"{f' . WarmRun[{true_i+1}/{warm_run}]' if true_i < warm_run else f' - Response {i - warm_run}/{run_number}'} - [{num}]{index+1}/{len(img_paths)}: Results # - {len(response)}")
                batch = list()

            #if (index+1)%2 == 0:
            #    break

        if true_i < warm_run:
            pass
        else:
            write_i = i - warm_run
            if write_i in write_main_indices:
                main_results = list()
                extended_main_results = list()
            time_results = list()
            for result in results:
                for instance in result:
                    if write_i in write_main_indices:
                        main_results.append(dict(
                            result=instance['result'],
                            origin_image_size=instance['origin_image_size'],
                            batch_image_size=instance['batch_image_size'],
                        ))
                        extended_main_results.extend(instance['result'])
                    time_s = dict(
                        preprocess_time=instance['preprocess_time'],
                        postprocess_time=instance['postprocess_time'],
                        inference_time=instance['inference_time'],
                    )
                    time_results.append(time_s)
                
            if write_i in write_main_indices:
                main_path = results_basepath.with_name(results_basepath.name + f".main.{write_i}")
                extended_main_path = results_basepath.with_name(results_basepath.name + f".main_extended.{write_i}")
            time_path = results_basepath.with_name(results_basepath.name + f".time.{write_i}")

            if write_i in write_main_indices:
                print(f"Write No.{write_i} MAIN results in File:\"{main_path}\".")
                main_results_json = json.dumps(main_results, indent=2)
                with open(main_path, 'w') as f:
                    f.write(main_results_json)

                print(f"Write No.{write_i} Ext MAIN results in File:\"{extended_main_path}\".")
                extended_main_results_json = json.dumps(extended_main_results, indent=2)
                with open(extended_main_path, 'w') as f:
                    f.write(extended_main_results_json)

            print(f"Write No.{write_i} Others results in File:\"{time_path}\".")
            time_results_json = json.dumps(time_results, indent=2)
            with open(time_path, 'w') as f:
                f.write(time_results_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multiple times on the Whole Test Dataset")
    parser.add_argument('--run-number', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--results-basepath', type=str, required=True)
    parser.add_argument('--model-path', type=str, default="../../bins/DETR_ResNet101_BSZ1.pth")
    parser.add_argument('--server-url', type=str, default="http://localhost:18080/predictions/DETR_ResNet101_BSZ1")
    parser.add_argument('--start-i', type=int, default=1)
    parser.add_argument('--write-main-indices', type=int, nargs='+', default=[1,])
    parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('--warm-run', type=int, default=1)

    # annotations should be placed in the --dataset-path.
    args = parser.parse_args()

    if args.local:
        model_path = Path(args.model_path)
        assert model_path.is_file(), f"Model Weights path {model_path.name} does not exist."

    dataset_root = Path(args.dataset_path)
    assert dataset_root.is_dir(), f"provided COCO path {dataset_root} does not exist"

    results_basepath = Path(args.results_basepath)
    assert results_basepath.parent.is_dir(), f"provided results saving dir {results_basepath.parent.name} does not exist."

    anno_path = os.path.join(dataset_root, "annotations", "instances_val2017.json")
    image_dir = os.path.join(dataset_root, "val2017")

    img_paths, img_ids = load_dataset(anno_path, image_dir)

    indices = set()
    for i in args.write_main_indices:
        if args.start_i <= i and i <= args.run_number:
            indices.add(i)

    if args.local:
        import torch
        from detr import DETR
        detr = DETR()
        state_dict = torch.load(args.model_path)
        detr.load_state_dict(state_dict, strict=True)
        detr.to('cuda:0')
        detr.eval()
        with torch.no_grad():
            run_all(args.start_i, args.run_number, args.batch_size, args.server_url, img_paths, img_ids, results_basepath, write_main_indices=indices, warm_run=args.warm_run, model=detr)
    else:
        run_all(args.start_i, args.run_number, args.batch_size, args.server_url, img_paths, img_ids, results_basepath, write_main_indices=indices, warm_run=args.warm_run)
