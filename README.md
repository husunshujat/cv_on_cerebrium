# CV on Cerebrium

A computer vision pipeline deployed on Cerebrium’s serverless AI infrastructure.


## 🎯 Getting Started

### Prerequisites

* Python 3.10
* Cerebrium CLI (`pip install cerebrium` + `cerebrium login`)
* Docker (for local builds)

### Quickstart

```bash
git clone https://github.com/husunshujat/cv_on_cerebrium.git
cd cv_on_cerebrium

# Install dependencies
pip install -r requirements.txt

# Convert the pth model to ONNX model:
python convert_to_onnx.py

# For testing:
#1. Add images to custom_test.py and their corresponding labels as two seperate lists. Then run
python test_server.py --customtest

#2. Directly just use image:
python test_server.py --image 'path/to/image'

#3. Check if cerebrium's api is working:
python test_server.py --health 'path/to/image'

## 🧑‍💻 License

MIT License — see [LICENSE](LICENSE)

--
