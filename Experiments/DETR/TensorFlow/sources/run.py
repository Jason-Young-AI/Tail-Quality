import os
import time
import json
import base64
import requests
import argparse

import tensorflow as tf

from PIL import Image
from pathlib import Path
from pycocotools.coco import COCO
from utils import read_jpeg_image, preprocess, cxcywh2xyxy, absolute2relative, xyxy2xywh


def pad_images_and_masks(images, masks):
    max_h = 0
    max_w = 0
    for image in images:
        h, w = list(image.shape[:2])
        max_h = max(max_h, h)
        max_w = max(max_w, w)
    
    padded_images = list()
    padded_masks = list()
    for image, mask in zip(images, masks):
        pad_h = max_h - tf.shape(mask)[0]
        pad_w = max_w - tf.shape(mask)[1]
        padded_images.append(tf.image.pad_to_bounding_box(image, 0, 0, max_h, max_w))

        paddings = [[0, pad_h], [0, pad_w]]
        padded_masks.append(tf.pad(mask, paddings, constant_values=True))
    
    padded_images = tf.stack(padded_images)
    padded_masks = tf.stack(padded_masks)
    return padded_images, padded_masks


def postprocess(outputs, image_sizes, image_indices):
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

    prob = tf.nn.softmax(out_logits, -1)[..., :-1]
    scores = tf.reduce_max(prob, axis=-1)
    labels = tf.argmax(prob, axis=-1)

    boxes = cxcywh2xyxy(out_bbox)

    results = list()
    for image_size, img_id, scores, labels, boxes in zip(image_sizes, image_indices, scores, labels, boxes):
        img_h, img_w = image_size
        sub_results = list()
        for score, label, box in zip(scores, labels, boxes):
            score = score.numpy()
            label = label.numpy()
            box = absolute2relative(box, (img_w, img_h))
            box = xyxy2xywh(box).numpy()
            sub_results.append({"image_id": img_id, "category_id": int(label), "bbox": box.tolist(), "score": float(score)})
        results.append(sub_results)

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
        raise NotImplementedError
        #images = dict()
        #for index, (img_path, img_id) in enumerate(batch):
        #    img = open(img_path, 'rb').read()
        #    img_str = base64.b64encode(img).decode('utf-8')
        #    images[index] = (img_str, img_id)

        #headers = {"Content-type": "application/json", "Accept": "text/plain"}
        #images = json.dumps(images)
        #response = requests.post(dest, data=images, headers=headers)
        #results, image_sizes, inference_time, pre_time, post_time = response.json()
    else:
        start_time = time.perf_counter()
        masks = list()
        images = list()
        image_sizes = list()
        image_indices = list()

        for index, (img_path, img_id) in enumerate(batch):
            raw_image = read_jpeg_image(img_path)
            h, w = list(raw_image.shape[:2])

            image, mask = preprocess(raw_image)

            masks.append(mask)
            images.append(image)
            image_sizes.append((h, w))
            image_indices.append(img_id)

        images, masks = pad_images_and_masks(images, masks)
        stop_time = time.perf_counter()
        pre_time = stop_time - start_time

        start_time = time.perf_counter()
        results = model((images, masks), training=False)
        stop_time = time.perf_counter()
        inference_time = stop_time - start_time

        start_time = time.perf_counter()
        results = postprocess(results, image_sizes, image_indices)
        stop_time = time.perf_counter()
        post_time = stop_time - start_time
        image_sizes = image_sizes

    results = add_other(results, image_sizes, inference_time, pre_time, post_time, images.shape[1:-1].as_list())
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
                main_path = args.results_basepath + f".main.{write_i}"
                extended_main_path = args.results_basepath + f".main_extended.{write_i}"
            time_path = args.results_basepath + f".time.{write_i}"

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
    parser.add_argument('--model-path', type=str, default="../bins/detr-r101-dc5-a2e86def.h5")
    parser.add_argument('--server-url', type=str, default="localhost:8500")
    parser.add_argument('--server-model', type=str, default="DETR_ResNet101")
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
    assert dataset_root.exists(), f"provided COCO path {dataset_root} does not exist"

    results_basepath = Path(args.results_basepath)
    assert results_basepath.parent.is_dir(), f"provided results saving dir {results_basepath.parent.name} does not exist."

    anno_path = os.path.join(dataset_root, "annotations", "instances_val2017.json")
    image_dir = os.path.join(dataset_root, "val2017")

    img_paths, img_ids = load_dataset(anno_path, image_dir)

    indices = set()
    for i in args.write_main_indices:
        if args.start_i <= i and i <= args.run_number:
            indices.add(i)

    assert args.batch_size in {1, 2}

    if args.batch_size == 1:
        dilation = True
        #params_path = f"../bins/detr-r101-dc5-a2e86def.h5"
    if args.batch_size == 2:
        dilation = False
        #params_path = f"../bins/detr-r101-2c7b67e5.h5"

    if args.local:
        from detr import DETR
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[:1], 'GPU')

        detr = DETR(dilation)
        detr.build()
        detr.load_weights(args.model_path)
        run_all(args.start_i, args.run_number, args.batch_size, args.server_url, img_paths, img_ids, results_basepath, write_main_indices=indices, warm_run=args.warm_run, model=detr)
    else:
        print("There is no implementation of Serve Mode!")
        #run_all(args.start_i, args.run_number, args.batch_size, args.server_url, img_paths, img_ids, write_main_indices=indices)
