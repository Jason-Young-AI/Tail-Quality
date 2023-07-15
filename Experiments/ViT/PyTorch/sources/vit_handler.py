import time
import base64
import torch

from PIL import Image
from torchvision import transforms

from ts.torch_handler.base_handler import BaseHandler


class ImageNetObjectDetector(BaseHandler):
    image_processing = transforms.Compose([
            #transforms.Resize(384, interpolation=str_to_interp_mode('bicubic')),
            #transforms.CenterCrop([384, 384]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor([0.5, 0.5, 0.5]),
                std=torch.tensor([0.5, 0.5, 0.5])
            )
        ])

    def preprocess(self, data_list):
        start_time = time.perf_counter()
        images = list()

        #if data.get("local", False):
        #    raw_images = data.get("local")
        #else:
        raw_images = data_list[0].get("data") or data_list[0].get("body")

        raw_image_indices = list(raw_images.keys())
        raw_image_indices.sort()

        for raw_image_index in raw_image_indices:
            raw_image_str = raw_images[raw_image_index]
            #assert isinstance(raw_image_str, str), f'Image Should be Recieved as Type \"str\" not: \"{type(raw_image_str)}\"'
            # if the image is a string of bytesarray.
            raw_image_bytes = base64.b64decode(raw_image_str)

            #assert isinstance(raw_image_bytes, (bytearray, bytes)), f'Image Should Decoded as Type \"byte(array, s)\" not: \"{type(raw_image_bytes)}\"'
            raw_image = Image.frombytes("RGB", (384, 384), raw_image_bytes)
            #raw_image = Image.open(io.BytesIO(raw_image_bytes)).convert("RGB")

            image = self.image_processing(raw_image)

            images.append(image)

        #print(f"Get {len(images)} Images To Predict.")
        images = torch.stack(images)

        self.batch_image_shape = images.shape[-2:]
        stop_time = time.perf_counter()

        self.pre_time = stop_time - start_time

        return images.to(self.device)#torch.stack(images).to(self.device)

    def postprocess(self, outputs):
        start_time = time.perf_counter()
        top5_probabilities, top5_class_indices = torch.topk(outputs.softmax(dim=1) * 100, k=5)

        #assert len(top5_class_indices) == len(self.image_sizes)
        #assert self.image_sizes.shape[1] == 2

        stop_time = time.perf_counter()

        self.post_time = stop_time - start_time

        return [(top5_class_indices.tolist(), top5_probabilities.tolist(), self.inference_time, self.pre_time, self.post_time, self.batch_image_shape), ]

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
    image = Image.open("../../beignets-task-guide.png")
    print(f"{image.size}")
    image = ImageNetObjectDetector.image_processing(image)
    print(f"{image.shape}")
