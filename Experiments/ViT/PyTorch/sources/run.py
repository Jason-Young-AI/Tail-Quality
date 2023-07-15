import time
import json
import torch
import base64
import requests
import argparse

from pathlib import Path
from PIL import Image
from torchvision import transforms
from imagenet import load_imagenet_val
from utils import str_to_interp_mode


image_preprocessing_local = transforms.Compose([
    transforms.Resize(384, interpolation=str_to_interp_mode('bicubic')),
    transforms.CenterCrop([384, 384]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=torch.tensor([0.5, 0.5, 0.5]),
        std=torch.tensor([0.5, 0.5, 0.5])
    )
])


image_preprocessing = transforms.Compose([
    transforms.Resize(384, interpolation=str_to_interp_mode('bicubic')),
    transforms.CenterCrop([384, 384]),
])


def postprocess(outputs):
    top5_probabilities, top5_class_indices = torch.topk(outputs.softmax(dim=1) * 100, k=5)
    return top5_class_indices, top5_probabilities


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
        images = dict()
        img_sizes = list()
        img_ids = list()
        for index, (img_path, img_id) in enumerate(batch):
            raw_image = Image.open(img_path).convert("RGB")
            img = image_preprocessing(raw_image)
            img = img.tobytes()
            img_str = base64.b64encode(img).decode('utf-8')
            images[index] = img_str
            img_ids.append(img_id)
            img_sizes.append(raw_image.size)

        #if model is not None:
        #    detector = ImageNetObjectDetector()
        #    detector.model = model
        #    payload = {'local': payload}
        #    payload = [payload,]
        #    payload = detector.preprocess(payload)
        #    payload = detector.inference(payload)
        #    payload = detector.postprocess(payload)
        #    return payload[0]
        #else:
        headers = {"Content-type": "application/json", "Accept": "text/plain"}
        images = json.dumps(images)
        response = requests.post(dest, data=images, headers=headers)
        top5_probabilities, top5_class_indices, inference_time, pre_time, post_time, batch_image_shape = response.json()
        results = [{'top5_probabilities': p, 'top5_class_indices': c} for p, c in zip(top5_probabilities, top5_class_indices)]
    else:
        start_time = time.perf_counter()
        images = list()
        img_sizes = list()
        img_ids = list()
        for index, (img_path, img_id) in enumerate(batch):
            raw_image = Image.open(img_path).convert("RGB")
            img = image_preprocessing_local(raw_image)
            images.append(img)
            img_ids.append(img_id)
            img_sizes.append(raw_image.size)

        #if model is not None:
        #    detector = ImageNetObjectDetector()
        #    detector.model = model
        #    payload = {'local': payload}
        #    payload = [payload,]
        #    payload = detector.preprocess(payload)
        #    payload = detector.inference(payload)
        #    payload = detector.postprocess(payload)
        #    return payload[0]
        #else:
        images = torch.stack(images).to('cuda:0')
        stop_time = time.perf_counter()
        pre_time = stop_time - start_time

        start_time = time.perf_counter()
        predictions = model(images)
        stop_time = time.perf_counter()
        inference_time = stop_time - start_time

        start_time = time.perf_counter()
        indices, values = postprocess(predictions)
        results = [{'top5_probabilities': prob, 'top5_class_indices': index} for index, prob in zip(indices.tolist(), values.tolist())]
        stop_time = time.perf_counter()
        post_time = stop_time - start_time
        batch_image_shape = images.shape[-2:]

    results = add_other(results, img_sizes, img_ids, inference_time, pre_time, post_time, batch_image_shape)
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

            #if (index + 1) % 512 == 0:
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
                            batch_image_size=instance['batch_image_size'],
                        ))
                    time_s = dict(
                        preprocess_time=instance['preprocess_time'],
                        postprocess_time=instance['postprocess_time'],
                        inference_time=instance['inference_time'],
                    )
                    time_results.append(time_s)
                
            if write_i in write_main_indices:
                main_path = results_basepath.with_name(results_basepath.name + f".main.{write_i}")
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
    parser.add_argument('--model-path', type=str, default="../bins/jx_vit_large_p32_384-9b920ba8.pth")
    parser.add_argument('--server-url', type=str, default="http://localhost:8080/predictions/ViT_L_P32_384")
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
    assert dataset_root.is_dir(), f"provided ImageNet2012 path {dataset_root.name} does not exist."

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
        import torch
        from vit import ViT
        vit = ViT()
        state_dict = torch.load(model_path)
        vit.load_state_dict(state_dict, strict=True)
        vit.to('cuda:0')
        vit.eval()
        with torch.no_grad():
            run_all(args.start_i, args.run_number, args.batch_size, args.server_url, img_paths, img_ids, results_basepath, write_main_indices=indices, warm_run=args.warm_run, model=vit)
    else:
        run_all(args.start_i, args.run_number, args.batch_size, args.server_url, img_paths, img_ids, results_basepath, write_main_indices=indices, warm_run=args.warm_run)
