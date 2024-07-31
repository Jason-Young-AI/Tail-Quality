import tensorflow.compat.v1 as tf
import time

from mobilenet.dataset import ImageNet, create_readable_names_for_imagenet_labels
from mobilenet.preprocessing import transform

batch_size = 100
#checkpoint = 'weights/mobilenet_v2_1.0_224.ckpt'
frozen_checkpoint = '/home/zxyang/MobileNet/weights/mobilenet_v2_1.0_224_frozen.pb'

tf.compat.v1.disable_eager_execution()
tf.reset_default_graph()

imagenet = ImageNet('~/MobileNet/datasets/val')

img_paths = [img_path for img_path, _ in imagenet]
label_ids = [label_id for _, label_id in imagenet]

image_ds = tf.data.Dataset.from_tensor_slices(img_paths)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(label_ids, tf.int64))

instance_ds = tf.data.Dataset.zip((image_ds, label_ds))
instance_ds = instance_ds.map(transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
instance_ds = instance_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
instance_it = tf.compat.v1.data.make_initializable_iterator(instance_ds)
next_batch = instance_it.get_next()

#image_ds = image_ds.map(transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#label_ds = label_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
#image_ds = image_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

#image_it = tf.compat.v1.data.make_initializable_iterator(image_ds)
#next_image_batch = image_it.get_next()

#label_it = tf.compat.v1.data.make_initializable_iterator(label_ds)
#next_label_batch = label_it.get_next()

#image_it = tf.compat.v1.data.make_initializable_iterator(image_ds)
#next_image_batch = image_it.get_next()

graphdef = tf.GraphDef.FromString(open(frozen_checkpoint, 'rb').read())
inputs, predictions = tf.import_graph_def(graphdef,  return_elements = ['input:0', 'MobilenetV2/Predictions/Reshape_1:0'])

with tf.Session(graph=inputs.graph) as session:
    for i in range(10):
        index = 0
        session.run(instance_it.initializer)
        # session.run(label_it.initializer)
        while True:
            try:
                image_batch, label_batch = session.run(next_batch)
                # label_batch = session.run(next_label_batch)
                index += 1
                a = time.perf_counter()
                x = predictions.eval(feed_dict={inputs: image_batch})
                b = time.perf_counter()
                print(f'{index}', b-a)
                #print(label_batch)
            except tf.errors.OutOfRangeError:
                break