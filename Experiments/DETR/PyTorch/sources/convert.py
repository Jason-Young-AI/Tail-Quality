import torch
from detr import DETR

detr = DETR()
state_dict = torch.load('/home/zxyang/HybridNet/DETR/DETR_ResNet101_BSZ1.pth')
detr.load_state_dict(state_dict, strict=True)
detr.to('cuda:0')
detr.eval()
dummy_input = torch.autograd.Variable(torch.randn(1, 3, 800, 1777)).to('cuda:0')

input_names = ['input_image']
output_names = ['pred_logits', 'pred_boxes']

dynamic_axes = {'input_image': {0: 'batch_size', 2: 'width', 3: 'height'}}

torch.onnx.export(detr, dummy_input, "model.onnx", input_names=input_names, output_names=output_names,
                    dynamic_axes=dynamic_axes, verbose=1, opset_version=11)
