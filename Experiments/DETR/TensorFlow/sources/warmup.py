import io
import os
import sys

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2

os.environ["CUDA_VISIBLE_DEVICES"] ="0"


# IMAGE_URLS are the locations of the images we use to warmup the model
shape = [1, 384, 384, 3]
dtype = tf.uint8

IMAGE_URLS = [
  tf.zeros(shape, dtype=dtype),
  tf.ones(shape, dtype=dtype)
]

# Current Resnet model in TF Model Garden (as of 7/2021) does not accept JPEG
# as input
MODEL_ACCEPT_JPG = False


def main():
  if len(sys.argv) != 2 or sys.argv[-1].startswith('-'):
    print('Usage: vit_warmup.py saved_model_dir')
    sys.exit(-1)

  model_dir = sys.argv[-1]
  if not os.path.isdir(model_dir):
    print('The saved model directory: %s does not exist. '
          'Specify the path of an existing model.' % model_dir)
    sys.exit(-1)

  # Create the assets.extra directory, assuming model_dir is the versioned
  # directory containing the SavedModel
  assets_dir = os.path.join(model_dir, 'assets.extra')
  if not os.path.exists(assets_dir):
    os.mkdir(assets_dir)

  warmup_file = os.path.join(assets_dir, 'tf_serving_warmup_requests')
  with tf.io.TFRecordWriter(warmup_file) as writer:
    for image in IMAGE_URLS:

      # Create the inference request
      request = predict_pb2.PredictRequest()
      request.model_spec.name = 'ViT_L_P32_384'
      request.model_spec.signature_name = 'serving_default'
      request.inputs['input_tensor'].CopyFrom(
          tf.make_tensor_proto(image))

      log = prediction_log_pb2.PredictionLog(
          predict_log=prediction_log_pb2.PredictLog(request=request))
      writer.write(log.SerializeToString())

  print('Created the file \'%s\', restart tensorflow_model_server to warmup '
        'the ViT SavedModel.' % warmup_file)

if __name__ == '__main__':
  main()
