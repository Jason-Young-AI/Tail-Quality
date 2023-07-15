import argparse

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU

from backbone import ResNet101Backbone
from transformer import Transformer
from custom_layers import Linear, FixedEmbedding
from position_embeddings import PositionEmbeddingSine
from utils import cxcywh2xyxy


class DETR(tf.keras.Model):
    def __init__(self, dilation=False):
        super().__init__()
        num_classes = 91
        num_queries = 100
        self.num_queries = num_queries

        self.backbone = ResNet101Backbone(replace_stride_with_dilation=[False, False, dilation], name='backbone')
        self.transformer = Transformer(return_intermediate_dec=True, name='transformer')
        self.model_dim = self.transformer.model_dim

        self.pos_encoder = PositionEmbeddingSine(num_pos_features=self.model_dim // 2, normalize=True)

        self.input_proj = Conv2D(self.model_dim, kernel_size=1, name='input_proj')

        self.query_embed = FixedEmbedding((num_queries, self.model_dim), name='query_embed')

        self.class_embed = Linear(num_classes + 1, name='class_embed')

        self.bbox_embed_linear1 = Linear(self.model_dim, name='bbox_embed_0')
        self.bbox_embed_linear2 = Linear(self.model_dim, name='bbox_embed_1')
        self.bbox_embed_linear3 = Linear(4, name='bbox_embed_2')
        self.activation = ReLU()

    def call(self, inp, training=False, post_process=False):
        x, masks = inp
        x = self.backbone(x, training=training)
        masks = self.downsample_masks(masks, x)
        pos_encoding = self.pos_encoder(masks)

        hs = self.transformer(self.input_proj(x), masks,
                              self.query_embed(None),
                              pos_encoding,
                              training=training)[0]

        outputs_class = self.class_embed(hs)

        box_ftmps = self.activation(self.bbox_embed_linear1(hs))
        box_ftmps = self.activation(self.bbox_embed_linear2(box_ftmps))
        outputs_coord = tf.sigmoid(self.bbox_embed_linear3(box_ftmps))

        output = {'pred_logits': outputs_class[-1],
                  'pred_boxes': outputs_coord[-1]}

        #return outputs_class[-1], outputs_coord[-1]
        return output

    def build(self, input_shape=None, **kwargs):
        if input_shape is None:
            input_shape = [(None, None, None, 3), (None, None, None)]
        super().build(input_shape, **kwargs)

    def downsample_masks(self, masks, x):
        masks = tf.cast(masks, tf.int32)
        masks = tf.expand_dims(masks, -1)
        # The existing tf.image.resize with method='nearest'
        # does not expose the half_pixel_centers option in TF 2.2.0
        # The original Pytorch F.interpolate uses it like this
        masks = tf.compat.v1.image.resize_nearest_neighbor(
            masks, tf.shape(x)[1:3], align_corners=False,
            half_pixel_centers=False)
        masks = tf.squeeze(masks, -1)
        masks = tf.cast(masks, tf.bool)
        return masks


#def get_serving_fn(model):
#
#    @tf.function(input_signature=[tf.TensorSpec(name="input_tensor", shape=[None, 384, 384, 3], dtype=tf.uint8)])
#    def serving_fn(raw_images):
#        #start_ts = tf.timestamp()
#        images = preprocess(raw_images)
#
#        #stop_ts = tf.timestamp()
#        #pre_ts = stop_ts - start_ts
#
#        predictions = model(images)
#
#        #start_ts = tf.timestamp()
#        topk_indices, topk_values = postprocess(predictions)
#        #stop_ts = tf.timestamp()
#        #post_ts = stop_ts - start_ts
#        #postprocessed_predictions = postprocess(predictions)
#        #return postprocessed_predictions
#        return topk_indices, topk_values#, inference_ts, pre_ts, post_ts
#
#    return serving_fn


#def save(dilation, params_path, saved_model_path):
#    detr = DETR(dilation)
#    detr.build()
#    detr.load_weights(params_path)
#    print(f"TensorFlow DETR Built!")
#
#    tf.saved_model.save(detr, saved_model_path, signatures={'serving_default': get_serving_fn(detr)})
#    print('Exporting trained model to', saved_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test or Convert Model Weights")
    parser.add_argument('--mode', required=True, choices=['create', 'test-tf', 'test-h5'])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--pth-path', type=str, default="../../bins/DETR_ResNet101_BSZ1.pth")
    parser.add_argument('--tf-path', type=str, default="../../bins/detr-r101-dc5-a2e86def.tf")
    parser.add_argument('--h5-path', type=str, default="../../bins/detr-r101-dc5-a2e86def.h5")
    args = parser.parse_args()
    #import os
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #if len(sys.argv) == 2 and sys.argv[1] == 'convert':
    #    tf.compat.v1.app.flags.DEFINE_integer('model_version', 1,
    #                                        'version number of the model.')
    #    FLAGS = tf.compat.v1.app.flags.FLAGS
    #    tf.compat.v1.disable_eager_execution()
    #    tf.compat.v1.app.run()
    # This produce wrong answer!!!!!!!!!!

    if args.batch_size == 1:
        dilation = True
    elif args.batch_size == 2:
        dilation = False
    else:
        raise ValueError

    if args.mode == 'create':
        raise NotImplementedError
        #save(dilation, args.pth_path, args.tf_path)

    elif args.mode == 'test-tf':
        raise NotImplementedError
        #detr = tf.saved_model.load(args.tf_path)
        #detr.build()
        #print("Tested TF")
    elif args.mode == 'test-h5':
        detr = DETR(dilation)
        detr.build()
        detr.load_weights(args.h5_path)
        print(f"Tested H5!")