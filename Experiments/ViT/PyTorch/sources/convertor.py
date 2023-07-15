# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch - TF 2.0 general utilities."""


def convert_pytorch_weights_to_tf2named_numpy(pt_state_dict):
    # Adapt pt state dict. TF "beta" -> PT "bias"
    # But some models have PT weight "beta" (ResMLP affine layer)
    # To fix that we need to change PT name to "bias" first...
    # Other models have PT weights "gamma" (ConvNeXt layer scale)
    old_keys = []
    new_keys = []
    for key in pt_state_dict.keys():
        new_key = None
        if key.endswith(".beta"):
            new_key = key.replace(".beta", ".bias")
        elif key.endswith(".gamma"):
            new_key = key.replace(".gamma", ".weight")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        pt_state_dict[new_key] = pt_state_dict.pop(old_key)

    all_pt_state_dict_keys = list(pt_state_dict.keys())
    for k in all_pt_state_dict_keys:
        # Find associated numpy array in pytorch model state dict
        pt_state_dict[k] = pt_state_dict[k].numpy()

    return pt_state_dict