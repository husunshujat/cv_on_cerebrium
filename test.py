import sys
import os
import numpy as np
from model import OnnxModelPredictor  # Your class file name here
from PIL import Image

def test_load_model(onnx_path):
    print(f"Testing loading ONNX model from: {onnx_path}")
    try:
        predictor = OnnxModelPredictor(onnx_path)
        print("✅ Model loaded successfully.")
        return predictor
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

def test_predict_with_pil(predictor, image_path):
    print(f"Testing prediction with PIL Image input: {image_path}")
    try:
        img = Image.open(image_path)
        output = predictor.predict(img)
        assert isinstance(output, np.ndarray), "Output is not a numpy array"
        assert output.ndim >= 2, "Output should have batch dimension and class scores"
        print(f"✅ Prediction successful. Output shape: {output.shape}")
        print(f"Predicted class index: {np.argmax(output)}")
    except Exception as e:
        print(f"❌ Prediction failed with PIL input: {e}")
        sys.exit(1)

def test_predict_with_numpy(predictor, image_path):
    print(f"Testing prediction with NumPy array input: {image_path}")
    try:
        img = Image.open(image_path)
        img_np = np.array(img)
        output = predictor.predict(img_np)
        assert isinstance(output, np.ndarray), "Output is not a numpy array"
        assert output.ndim >= 2, "Output should have batch dimension and class scores"
        print(f"✅ Prediction successful with NumPy input. Output shape: {output.shape}")
        print(f"Predicted class index: {np.argmax(output)}")
    except Exception as e:
        print(f"❌ Prediction failed with NumPy input: {e}")
        sys.exit(1)

def test_invalid_input(predictor):
    print("Testing prediction with invalid input type")
    try:
        predictor.predict("this is not an image")
        print("❌ Error: Model should have raised exception for invalid input")
        sys.exit(1)
    except TypeError as e:
        print(f"✅ Properly raised TypeError for invalid input. e:{e}")
    except Exception as e:
        print(f"❌ Unexpected exception type: {e}")
        sys.exit(1)

def main():
    ONNX_MODEL_PATH = "model_with_preprocessing.onnx"  # Adjust if needed
    SAMPLE_IMAGE_PATH = "n01667114_mud_turtle.JPEG"  # Adjust if needed

    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"❌ ONNX model file not found at {ONNX_MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(SAMPLE_IMAGE_PATH):
        print(f"❌ Sample image file not found at {SAMPLE_IMAGE_PATH}")
        sys.exit(1)

    predictor = test_load_model(ONNX_MODEL_PATH)
    test_predict_with_pil(predictor, SAMPLE_IMAGE_PATH)
    test_predict_with_numpy(predictor, SAMPLE_IMAGE_PATH)
    test_invalid_input(predictor)

    print("\n🎉 All tests passed successfully!")

if __name__ == "__main__":
    main()
