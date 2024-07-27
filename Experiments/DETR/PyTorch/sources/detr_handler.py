import io
import time
import base64
import torch
import torch.nn.functional as F

from PIL import Image

from transforms import Compose, DetermineResize, ToTensor, Normalize
from box_ops import box_cxcywh_to_xyxy
from utils import prepare_for_coco_detection
from misc import nested_tensor_from_tensor_list

from ts.torch_handler.base_handler import BaseHandler


class MSCoCoObjectDetector(BaseHandler):
    image_processing = Compose([
            DetermineResize(800, max_size=1333),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def preprocess(self, data_list):
        start_time = time.perf_counter()
        images = list()
        image_sizes = list()
        image_indices = list()

        raw_images = data_list[0].get("data") or data_list[0].get("body")

        raw_image_indices = list(raw_images.keys())
        raw_image_indices.sort()

        for raw_image_index in raw_image_indices:
            raw_image_str, image_index = raw_images[raw_image_index]
            #assert isinstance(raw_image_str, str), f'Image Should be Recieved as Type \"str\" not: \"{type(raw_image_str)}\"'
            # if the image is a string of bytesarray.
            raw_image_bytes = base64.b64decode(raw_image_str)

            #assert isinstance(raw_image_bytes, (bytearray, bytes)), f'Image Should Decoded as Type \"byte(array, s)\" not: \"{type(raw_image_bytes)}\"'
            raw_image = Image.open(io.BytesIO(raw_image_bytes)).convert("RGB")

            w, h = raw_image.size

            image = self.image_processing(raw_image)

            images.append(image)
            image_sizes.append(torch.as_tensor([int(h), int(w)]))
            image_indices.append(image_index)

        #print(f"Get {len(images)} Images To Predict.")
        images = nested_tensor_from_tensor_list(images)
        self.batch_image_shape = images.tensors.shape[-2:]
        stop_time = time.perf_counter()
        self.pre_time = stop_time - start_time

        self.image_sizes = torch.stack(image_sizes, dim=0)
        self.image_indices = image_indices
        #print(f"Batch Size: {self.batch_size}")
        #print(f"Original Image Sizes: {self.image_sizes}")

        return images.to(self.device)#torch.stack(images).to(self.device)

    def postprocess(self, outputs):
        start_time = time.perf_counter()
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        #assert len(out_logits) == len(self.image_sizes)

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = self.image_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scale_fct = scale_fct.to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        raw_results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        results = prepare_for_coco_detection(raw_results, self.image_indices)

        stop_time = time.perf_counter()
        self.post_time = stop_time - start_time

        return [(results, self.image_sizes.tolist(), self.inference_time, self.pre_time, self.post_time, self.batch_image_shape), ]

    def inference(self, data, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.

        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.

        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """
        with torch.no_grad():
            marshalled_data = data.to(self.device)
            start_time = time.perf_counter()
            results = self.model(marshalled_data, *args, **kwargs)
            stop_time = time.perf_counter()
        self.inference_time = stop_time - start_time
        return results


if __name__ == "__main__":
    image = Image.open("/home/zxyang/1.Datasets/coco2017/val2017/000000581781.jpg").convert("RGB")
    print(f"{image.size}")
    image = MSCoCoObjectDetector.image_processing(image)
    print(f"{image.shape}")
