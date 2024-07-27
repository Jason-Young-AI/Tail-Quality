import onnx
from onnx_tf.backend import prepare

model = onnx.load('model.onnx')

# Import the ONNX model to Tensorflow
tf_rep = prepare(model)

# don't use the ".pb" in the name of the exported file, so that it creates a proper folder for the weights
tf_rep.export_graph("detr")
