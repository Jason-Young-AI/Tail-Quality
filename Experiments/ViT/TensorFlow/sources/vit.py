"""
TensorFlow implementation of the Visual Transformer

Based on timm/models/visual_transformer.py by Ross Wightman.
Based on transformers/models/vit by HuggingFace

Copyright 2021 Martins Bruveris
Modified by XXYYZZ
"""
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import os
import sys
import argparse
import tensorflow as tf

from layer_factory import norm_layer_factory
from drop_layers import DropPath
from transformer_layers import MLP, PatchEmbeddings, interpolate_pos_embeddings
from data_process import preprocess, postprocess
#from tensorflow.python.ops.numpy_ops import np_config

#np_config.enable_numpy_behavior()


IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
os.environ["CUDA_VISIBLE_DEVICES"] ="0"


@dataclass
class ViTConfig():
    nb_classes: int = 1000
    in_channels: int = 3
    input_size: Tuple[int, int] = (384, 384)
    patch_layer: str = "patch_embeddings"
    patch_nb_blocks: tuple = ()
    patch_size: int = 32
    embed_dim: int = 1024
    nb_blocks: int = 24
    nb_heads: int = 16
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    representation_size: Optional[int] = None
    distilled: bool = False
    # Regularization
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # Other parameters
    norm_layer: str = "layer_norm_eps_1e-6"
    act_layer: str = "gelu"
    # Parameters for inference
    interpolate_input: bool = False
    crop_pct: float = 1
    interpolation: str = "bicubic"
    mean: Tuple[float, float, float] = IMAGENET_INCEPTION_MEAN
    std: Tuple[float, float, float] = IMAGENET_INCEPTION_STD
    first_conv: str = "patch_embed/proj"
    # DeiT models have two classifier heads, one for distillation
    classifier: Union[str, Tuple[str, str]] = "head"

    """
    Args:
        nb_classes: Number of classes for classification head
        in_channels: Number of input channels
        input_size: Input image size
        patch_layer: Layer used to transform image to patches. Possible values are
            `patch_embeddings` and `hybrid_embeddings`.
        patch_nb_blocks: When `patch_layer="hybrid_embeddings`, this is the number of
            residual blocks in each stage. Set to `()` to use only the stem.
        patch_size: Patch size; Image size must be multiple of patch size. For hybrid
            embedding layer, this patch size is applied after the convolutional layers.
        embed_dim: Embedding dimension
        nb_blocks: Depth of transformer (number of encoder blocks)
        nb_heads: Number of self-attention heads
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        qkv_bias: Enable bias for qkv if True
        representation_size: Enable and set representation layer (pre-logits) to this
            value if set
        distilled: Model includes a distillation token and head as in DeiT models
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Dropout rate for stochastic depth
        norm_layer: Normalization layer
        act_layer: Activation function
    """

    @property
    def nb_tokens(self) -> int:
        """Number of special tokens"""
        return 2 if self.distilled else 1

    @property
    def grid_size(self) -> Tuple[int, int]:
        grid_size = (
            self.input_size[0] // self.patch_size,
            self.input_size[1] // self.patch_size,
        )
        if self.patch_layer == "hybrid_embeddings":
            # 2 reductions in the stem, 1 reduction in each stage except the first one
            reductions = 2 + max(len(self.patch_nb_blocks) - 1, 0)
            stride = 2**reductions
            grid_size = (grid_size[0] // stride, grid_size[1] // stride)
        return grid_size

    @property
    def nb_patches(self) -> int:
        """Number of patches without class and distillation tokens."""
        return self.grid_size[0] * self.grid_size[1]

    @property
    def transform_weights(self):
        return {"pos_embed": ViT.transform_pos_embed}


class ViTMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.nb_heads = nb_heads
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate

        head_dim = embed_dim // nb_heads
        self.scale = head_dim**-0.5

        self.qkv = tf.keras.layers.Dense(
            units=3 * embed_dim, use_bias=qkv_bias, name="qkv"
        )
        self.attn_drop = tf.keras.layers.Dropout(rate=attn_drop_rate)
        self.proj = tf.keras.layers.Dense(units=embed_dim, name="proj")
        self.proj_drop = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training=False, return_features=False):
        features = OrderedDict()

        # B (batch size), N (sequence length), D (embedding dimension),
        # H (number of heads)
        batch_size, seq_length = tf.unstack(tf.shape(x)[:2])
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = tf.reshape(qkv, (batch_size, seq_length, 3, self.nb_heads, -1))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, H, N, D/H)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = self.scale * tf.linalg.matmul(q, k, transpose_b=True)  # (B, H, N, N)
        attn = tf.nn.softmax(attn, axis=-1)  # (B, H, N, N)
        attn = self.attn_drop(attn, training=training)
        features["attn"] = attn

        x = tf.linalg.matmul(attn, v)  # (B, H, N, D/H)
        x = tf.transpose(x, (0, 2, 1, 3))  # (B, N, H, D/H)
        x = tf.reshape(x, (batch_size, seq_length, -1))  # (B, N, D)

        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return (x, features) if return_features else x


class ViTBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        norm_layer: str,
        act_layer: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.nb_heads = nb_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        norm_layer = norm_layer_factory(norm_layer)

        self.norm1 = norm_layer(name="norm1")
        self.attn = ViTMultiHeadAttention(
            embed_dim=embed_dim,
            nb_heads=nb_heads,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            name="attn",
        )
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.norm2 = norm_layer(name="norm2")
        self.mlp = MLP(
            hidden_dim=int(embed_dim * mlp_ratio),
            embed_dim=embed_dim,
            drop_rate=drop_rate,
            act_layer=act_layer,
            name="mlp",
        )

    def call(self, x, training=False, return_features=False):
        features = OrderedDict()
        shortcut = x
        x = self.norm1(x, training=training)
        x = self.attn(x, training=training, return_features=return_features)
        if return_features:
            x, mha_features = x
            features["attn"] = mha_features["attn"]
        x = self.drop_path(x, training=training)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x, training=training)
        x = self.mlp(x, training=training)
        x = self.drop_path(x, training=training)
        x = x + shortcut
        return (x, features) if return_features else x


class ViT(tf.keras.Model):
    def __init__(self):
        super().__init__()

        cfg = ViTConfig()

        self.cfg = cfg
        self.nb_features = cfg.embed_dim  # For consistency with other models
        self.norm_layer = norm_layer_factory(cfg.norm_layer)

        self.patch_embed = PatchEmbeddings(
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            norm_layer="",  # ViT does not use normalization in patch embeddings
            name="patch_embed",
        )
        self.cls_token = None
        self.dist_token = None
        self.pos_embed = None
        self.pos_drop = tf.keras.layers.Dropout(rate=cfg.drop_rate)

        self.blocks = [
            ViTBlock(
                embed_dim=cfg.embed_dim,
                nb_heads=cfg.nb_heads,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                drop_rate=cfg.drop_rate,
                attn_drop_rate=cfg.attn_drop_rate,
                drop_path_rate=cfg.drop_path_rate,
                norm_layer=cfg.norm_layer,
                act_layer=cfg.act_layer,
                name=f"blocks/{j}",
            )
            for j in range(cfg.nb_blocks)
        ]
        self.norm = self.norm_layer(name="norm")

        self.pre_logits = None

        # Classifier head(s)
        self.head = (
            tf.keras.layers.Dense(units=cfg.nb_classes, name="head")
            if cfg.nb_classes > 0
            else tf.keras.layers.Activation("linear")  # Identity layer
        )
        self.head_dist = None

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.cfg.embed_dim),
            initializer="zeros",
            trainable=True,
            name="cls_token",
        )
        self.dist_token = (
            self.add_weight(
                shape=(1, 1, self.cfg.embed_dim),
                initializer="zeros",
                trainable=True,
                name="dist_token",
            )
            if self.cfg.distilled
            else None
        )
        self.pos_embed = self.add_weight(
            shape=(1, self.cfg.nb_patches + self.cfg.nb_tokens, self.cfg.embed_dim),
            initializer="zeros",
            trainable=True,
            name="pos_embed",
        )

    @property
    def dummy_inputs(self) -> tf.Tensor:
        return tf.zeros((1, *self.cfg.input_size, self.cfg.in_channels))

    @property
    def feature_names(self) -> List[str]:
        """
        Names of features, returned when calling ``call`` with ``return_features=True``.
        """
        _, features = self(self.dummy_inputs, return_features=True)
        return list(features.keys())

    def transform_pos_embed(self, src_weights, target_cfg: ViTConfig):
        return interpolate_pos_embeddings(
            pos_embed=self.pos_embed,
            src_grid_size=self.cfg.grid_size,
            tgt_grid_size=target_cfg.grid_size,
            nb_tokens=self.cfg.nb_tokens,
        )

    def forward_features(self, x, training=False, return_features=False):
        features = OrderedDict()
        batch_size = tf.shape(x)[0]

        x, grid_size = self.patch_embed(x, return_shape=True)
        cls_token = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        if not self.cfg.distilled:
            x = tf.concat((cls_token, x), axis=1)
        else:
            dist_token = tf.repeat(self.dist_token, repeats=batch_size, axis=0)
            x = tf.concat((cls_token, dist_token, x), axis=1)
        if not self.cfg.interpolate_input:
            x = x + self.pos_embed
        else:
            pos_embed = interpolate_pos_embeddings(
                self.pos_embed,
                src_grid_size=self.cfg.grid_size,
                tgt_grid_size=grid_size,
                nb_tokens=self.cfg.nb_tokens,
            )
            x = x + pos_embed
        x = self.pos_drop(x, training=training)
        features["patch_embedding"] = x

        for j, block in enumerate(self.blocks):
            x = block(x, training=training, return_features=return_features)
            if return_features:
                x, block_features = x
                features[f"block_{j}/attn"] = block_features["attn"]
            features[f"block_{j}"] = x
        x = self.norm(x, training=training)
        features["features_all"] = x

        if self.cfg.distilled:
            # Here we diverge from timm and return both outputs as one tensor. That way
            # all models always have one output by default
            x = x[:, :2]
        elif self.cfg.representation_size:
            x = self.pre_logits(x[:, 0])
        else:
            x = x[:, 0]
        features["features"] = x
        return (x, features) if return_features else x

    def call(self, x, training=False, return_features=False):
        #start_ts = tf.timestamp()
        features = {}
        x = self.forward_features(x, training, return_features)
        if return_features:
            x, features = x
        if not self.cfg.distilled:
            x = self.head(x)
        else:
            y = self.head(x[:, 0])
            y_dist = self.head_dist(x[:, 1])
            x = tf.stack((y, y_dist), axis=1)
        features["logits"] = x
        #stop_ts = tf.timestamp()
        #inference_ts = stop_ts - start_ts
        return (x, features) if return_features else x#, inference_ts


def get_serving_fn(model):

    @tf.function(input_signature=[tf.TensorSpec(name="input_tensor", shape=[None, 384, 384, 3], dtype=tf.uint8)])
    def serving_fn(raw_images):
        #start_ts = tf.timestamp()
        images = preprocess(raw_images)

        #stop_ts = tf.timestamp()
        #pre_ts = stop_ts - start_ts

        predictions = model(images)

        #start_ts = tf.timestamp()
        topk_indices, topk_values = postprocess(predictions)
        #stop_ts = tf.timestamp()
        #post_ts = stop_ts - start_ts
        #postprocessed_predictions = postprocess(predictions)
        #return postprocessed_predictions
        return topk_indices, topk_values#, inference_ts, pre_ts, post_ts

    return serving_fn


def save(model_path, h5_path, tf_path):
    import numpy
    loaded = numpy.load(model_path, allow_pickle=True)
    loaded = loaded[()]
    print(f"TF2 named NumPy state_dict loaded.")

    vit = ViT()
    vit(vit.dummy_inputs, training=False)
    print(f"TensorFlow ViT Built!")

    from convertor import load_pytorch_weights_in_tf2_model
    load_pytorch_weights_in_tf2_model(vit, loaded)
    print(f"Conversion Finished! PyTorch ==>> TensorFlow.")

    #vit.save(export_path)
    tf.saved_model.save(vit, tf_path, signatures={'serving_default': get_serving_fn(vit)})
    #tf.saved_model.save(vit, export_path, clear_devices=True)
    #vit.save_weights(save_path)
    print('Exporting trained model to', tf_path)
    #builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)
    #builder.save()

    vit.save_weights(h5_path)
    print('Model saved to', h5_path)


    #import numpy
    #loaded = numpy.load("../bins/jx_vit_large_p32_tf2named_numpy_state_dict.npy", allow_pickle=True)
    #loaded = loaded[()]
    #print(f"TF2 named NumPy state_dict loaded.")

    #vit = ViT()
    #vit(vit.dummy_inputs, training=False)
    #print(f"TensorFlow ViT Built!")

    #from convertor import load_pytorch_weights_in_tf2_model
    #load_pytorch_weights_in_tf2_model(vit, loaded)
    #print(f"Conversion Finished! PyTorch ==>> TensorFlow.")

    #save_path = "../bins/jx_vit_large_p32_tf2.h5"
    ##vit.save(export_path)
    #vit.save_weights(save_path)
    #print('Model saved to', save_path)
    ##builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)
    ##builder.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test or Convert Model Weights")
    parser.add_argument('--mode', required=True, choices=['convert', 'test-tf', 'test-h5', 'test'])
    parser.add_argument('--model-path', type=str, default="../bins/jx_vit_large_p32_tf2named_numpy_state_dict.npy")
    parser.add_argument('--h5-path', type=str, default="../bins/jx_vit_large_p32_tf2.h5")
    parser.add_argument('--tf-path', type=str, default="../bins/jx_vit_large_p32_tf2.tf/1/")
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

    if args.mode == 'convert':
        save(args.model_path, args.h5_path, args.tf_path)

    elif args.mode == 'test-tf':
        vit = tf.saved_model.load(args.tf_path)
        test_inputs = x = tf.zeros((1, *[384, 384], 3))
        print(x.shape)
        print(vit(test_inputs).shape)
        print("Tested TF")
    elif args.mode == 'test-h5':
        vit = ViT()
        vit(vit.dummy_inputs, training=False)
        vit.load_weights(args.h5_path)
        vit(vit.dummy_inputs, training=False)
        print(f"Tested H5!")
    elif args.mode == 'test':
        vit = ViT()
        vit(vit.dummy_inputs, training=False)
        print(f"TensorFlow ViT Built!")