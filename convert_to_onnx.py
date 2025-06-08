from pytorch_model import Classifier, BasicBlock, WrappedModel
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# Load base model
base_model = Classifier(BasicBlock, [2, 2, 2, 2])
base_model.load_state_dict(torch.load("../mtailor/pytorch_model_weights.pth"))
base_model.eval()

# Wrap with preprocessing
model = WrappedModel(base_model)
model.eval()

# Dummy input as raw image (uint8 or float32 unnormalized)
dummy_input = torch.randint(0, 255, (1, 3, 256, 256), dtype=torch.uint8)  # e.g. raw image

# Export
torch.onnx.export(
    model,
    dummy_input,
    "model_with_preprocessing.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
print("ONNX Model Exported: model_with_preprocessing.onnx")