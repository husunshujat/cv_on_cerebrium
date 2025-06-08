import numpy as np
from PIL import Image
from model import OnnxModelPredictor


def run(image_file_path: str, run_id=None):
    """
    param_1: Image file path (string)
    run_id: Optional, injected by Cerebrium

    Returns:
        dict with predicted class and status code
    """
    try:
        predictor = OnnxModelPredictor('model_with_preprocessing.onnx')
        img = Image.open(image_file_path)
        output = predictor.predict(img)
        predicted_class = int(np.argmax(output))

        return {
            "predicted_class": predicted_class,
            "status_code": 200
        }
    except Exception as e:
        return {
            "error": str(e),
            "status_code": 500
        }