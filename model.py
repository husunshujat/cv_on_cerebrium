import onnxruntime as ort
import numpy as np
from PIL import Image as PILImage
from torchvision.transforms import ToTensor
import onnx


class OnnxModelPredictor:
    def __init__(self, onnx_model_path: str):
        self.session = ort.InferenceSession(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        if isinstance(img, PILImage.Image):
            img = img.convert("RGB")
            img = img.resize((256, 256))  # Match the ONNX export input shape
            img_np = np.array(img).astype(np.uint8)
        elif isinstance(img, np.ndarray):
            img_np = img.astype(np.uint8)
            if img_np.shape[:2] != (256, 256):
                img_np = np.array(PILImage.fromarray(img_np).resize((256, 256)))
        else:
            raise TypeError("Input must be PIL.Image or numpy.ndarray")

        img_np = np.transpose(img_np, (2, 0, 1))  # HWC to CHW
        input_tensor = np.expand_dims(img_np, axis=0).copy()  # Add batch dim and ensure contiguous

        return input_tensor

    def predict(self, img):
        """
        Predict using the ONNX model.
        img: PIL.Image or numpy.ndarray (H,W,C)
        Returns: output numpy array (model raw output)
        """
        input_tensor = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return outputs[0]
