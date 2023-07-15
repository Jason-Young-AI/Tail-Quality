import json
import grpc
import time
import numpy
import base64
import argparse
#import requests

import tensorflow as tf

from PIL import Image
from pathlib import Path
from imagenet import load_imagenet_val
from data_process import compute_resized_output_size, center_crop, preprocess, postprocess

from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub


IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


def add_other(imagenet_results, image_sizes, image_indices, inference_time, pre_time, post_time, image_shape):
    imagenet_results_with_other = list()
    for imagenet_result, image_size, image_index in zip(imagenet_results, image_sizes, image_indices):
        imagenet_results_with_other.append(
            dict(
                origin_image_size=image_size,
                batch_image_size=image_shape,
                result=imagenet_result,
                image_id=image_index,
                inference_time=inference_time,
                postprocess_time=post_time,
                preprocess_time=pre_time,
            )
        )

    return imagenet_results_with_other


def req_post(dest, batch, model=None):
    if model is None:
        start_time = time.perf_counter()
        #payload = dict()
        images = list()
        img_sizes = list()
        img_ids = list()
        for index, (img_path, img_id) in enumerate(batch):
            raw_image = Image.open(img_path).convert("RGB")
            w, h = raw_image.size
            target_height, target_width = compute_resized_output_size((h, w), 384)
            img = raw_image.resize(tuple([target_width, target_height]), Image.BICUBIC)
            img = tf.convert_to_tensor(numpy.array(img), dtype="uint8")
            img = center_crop(img, (384, 384))
            images.append(img)
            img_ids.append(img_id)
            img_sizes.append(raw_image.size)
        #img_str = base64.b64encode(img).decode('utf-8')
        #payload[index] = img_str

        images = tf.stack(images)
        channel = grpc.insecure_channel(dest[0])
        service_stub = PredictionServiceStub(channel)

        imgs_tensor = tf.make_tensor_proto(images, shape=images.shape)
        grpc_request = PredictRequest()
        grpc_request.model_spec.name = dest[1]
        grpc_request.model_spec.signature_name = "serving_default"
        grpc_request.inputs["input_tensor"].CopyFrom(imgs_tensor)
        #payload = {'raw_images': payload}
        #headers = {"Content-type": "application/json", "Accept": "text/plain"}
        #payload = json.dumps(payload)
        #response = requests.post(dest, data=payload, headers=headers)
        stop_time = time.perf_counter()
        pre_time = stop_time - start_time

        start_time = time.perf_counter()
        response = service_stub.Predict(grpc_request, 20.0)  # 10 sec timeout
        stop_time = time.perf_counter()
        inference_time = stop_time - start_time

        start_time = time.perf_counter()
        indices = tf.make_ndarray(response.outputs["output_0"])
        values = tf.make_ndarray(response.outputs["output_1"])
        results = [{"top5_class_indices": index, "top5_probabilities": prob} for index, prob in zip(indices.tolist(), values.tolist())]
        stop_time = time.perf_counter()
        post_time = stop_time - start_time
    else:
        #payload = dict()
        start_time = time.perf_counter()
        images = list()
        img_sizes = list()
        img_ids = list()
        for index, (img_path, img_id) in enumerate(batch):
            raw_image = Image.open(img_path).convert("RGB")
            w, h = raw_image.size
            target_height, target_width = compute_resized_output_size((h, w), 384)
            img = raw_image.resize(tuple([target_width, target_height]), Image.BICUBIC)
            img = tf.convert_to_tensor(numpy.array(img), dtype="uint8")
            img = center_crop(img, (384, 384))
            img = preprocess(img)

            images.append(img)
            img_ids.append(img_id)
            img_sizes.append(raw_image.size)
        #img_str = base64.b64encode(img).decode('utf-8')
        #payload[index] = img_str

        images = tf.stack(images)
        stop_time = time.perf_counter()
        pre_time = stop_time - start_time

        start_time = time.perf_counter()
        predictions = model(images, training=False)
        stop_time = time.perf_counter()
        inference_time = stop_time - start_time

        start_time = time.perf_counter()
        indices, values = postprocess(predictions)
        #indices, values, inference_time, pre_time, post_time = predict(images)
        results = [{"top5_class_indices": index, "top5_probabilities": prob} for index, prob in zip(indices.numpy().tolist(), values.numpy().tolist())]
        stop_time = time.perf_counter()
        post_time = stop_time - start_time
    results = add_other(results, img_sizes, img_ids, inference_time, pre_time, post_time, images.shape[1:-1].as_list())
    return results


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

            #if (index + 1) % 512 ==0:
            #    break

        if true_i < warm_run:
            pass
        else:
            write_i = i - warm_run
            if write_i in write_main_indices:
                main_results = list()
            time_results = list()
            for result in results:
                for instance in result:
                    if write_i in write_main_indices:
                        main_results.append(dict(
                            result=instance['result'],
                            image_id=instance['image_id'],
                            origin_image_size=instance['origin_image_size'],
                            batch_image_size=instance['batch_image_size']
                        ))
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
    parser.add_argument('--model-path', type=str, default="../bins/jx_vit_large_p32_tf2.h5")
    parser.add_argument('--server-url', type=str, default="localhost:8500")
    parser.add_argument('--server-model', type=str, default="ViT_L_P32_384")
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
    assert dataset_root.exists(), f"provided ImageNet2012 path {dataset_root} does not exist."

    results_basepath = Path(args.results_basepath)
    assert results_basepath.parent.is_dir(), f"provided results saving dir {results_basepath.parent.name} does not exist."

    imagenet_val_instances = load_imagenet_val(dataset_root)

    img_paths = list()
    img_ids = list()
    for img_path, img_id in imagenet_val_instances:
        img_paths.append(img_path)
        img_ids.append(img_id)

    indices = set()
    for i in args.write_main_indices:
        if args.start_i <= i and i <= args.run_number:
            indices.add(i)

    if args.local:
        from vit import ViT
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[:1], 'GPU')
        vit = ViT()
        vit(vit.dummy_inputs, training=False)
        vit.load_weights(model_path)
        run_all(args.start_i, args.run_number, args.batch_size, args.server_url, img_paths, img_ids, results_basepath, write_main_indices=indices, warm_run=args.warm_run, model=vit)
    else:
        server_url = (args.server_url, args.server_model)
        tf.config.set_visible_devices([], 'GPU')
        run_all(args.start_i, args.run_number, args.batch_size, server_url, img_paths, img_ids, results_basepath, write_main_indices=indices, warm_run=args.warm_run)
