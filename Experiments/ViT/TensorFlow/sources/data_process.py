import tensorflow as tf

IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


def compute_resized_output_size(image_size, size):
    h, w = image_size
    short, long = (w, h) if w <= h else (h, w)
    requested_new_short = size

    new_short, new_long = requested_new_short, int(requested_new_short * long / short)

    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    return [new_h, new_w]


def center_crop(image, crop_size):
    image_height = int(image.shape[0])
    image_width = int(image.shape[1])
    #image_height = tf.cast(image.shape[0], dtype=tf.uint8)
    #image_width = tf.cast(image.shape[1], dtype=tf.uint8)
    crop_height, crop_width = crop_size

    if crop_width > image_width or crop_height > image_height:
        offset_height = (crop_height - image_height) // 2 if crop_height > image_height else 0,
        offset_width = (crop_width - image_width) // 2 if crop_width > image_width else 0,
        image = tf.image.pad_to_bounding_box(image, offset_height, offset_width, crop_height, crop_width)
        #image_height = tf.cast(image.shape[0], dtype=tf.uint8)
        #image_width = tf.cast(image.shape[1], dtype=tf.uint8)
        image_height = int(image.shape[0])
        image_width = int(image.shape[1])
        if crop_width == image_width and crop_height == image_height:
            return image

    #crop_top = tf.cast(tf.math.round((image_height - crop_height) / 2.0), dtype=tf.uint8)
    #crop_left = tf.cast(tf.math.round((image_width - crop_width) / 2.0), dtype=tf.uint8)
    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))

    cropped_image = tf.image.crop_to_bounding_box(
        image,
        crop_top,
        crop_left,
        crop_height,
        crop_width
    )

    return cropped_image


def normalize_img(img):
    # Scale to the value range of [0, 1] first and then normalize.
    dtype = "float32"
    mean = tf.convert_to_tensor([0.5, 0.5, 0.5], dtype=dtype)
    std = tf.convert_to_tensor([0.5, 0.5, 0.5], dtype=dtype)

    img = (img - mean) / std
    return img


def preprocess(raw_images):
    #img = raw_image.resize(tuple([target_width, target_height]), Image.BICUBIC)
    #img = tf.convert_to_tensor(numpy.array(img), dtype="uint8")
    ##img = tf.image.resize(img, size=tuple(compute_resized_output_size((h, w), 384)), method=tf.image.ResizeMethod.BICUBIC)
    ##img = tf.image.resize_with_pad(img, target_height, target_width, method=tf.image.ResizeMethod.BICUBIC)
    #img = center_crop(raw_image, (384, 384))
    imgs = tf.cast(raw_images, dtype="float32") / 255.0
    #img = normalize_img(img)
    mean = tf.constant([0.5, 0.5, 0.5], dtype="float32")
    std = tf.constant([0.5, 0.5, 0.5], dtype="float32")

    imgs = (imgs - mean) / std
    return imgs


def postprocess(predictions):
    #indices = tf.argmax(predictions, axis=1).numpy().tolist()
    probs = tf.nn.softmax(predictions, axis=1) * 100
    #pred_confidence = tf.reduce_max(probs, axis=1).numpy().tolist()
    topk = tf.math.top_k(probs, k=5)
    #results = [{"top5_class_indices": index, "top5_probabilities": prob} for index, prob in zip(topk.indices.numpy().tolist(), topk.values.numpy().tolist())]
    #results = [{"top5_class_indices": index, "top5_probabilities": prob} for index, prob in zip(topk.indices, topk.values)]

    return topk.indices, topk.values