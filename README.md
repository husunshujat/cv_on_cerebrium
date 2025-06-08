# CV on Cerebrium

A computer vision pipeline deployed on Cerebrium’s serverless AI infrastructure.

## 🚀 Overview

**cv\_on\_cerebrium** showcases how to build, train, and deploy machine learning models—especially in computer vision—using Cerebrium’s platform. This repo provides tools and examples to:

* Train and evaluate CV models locally or on Cerebrium
* Package models into scalable serverless endpoints
* Run inference via RESTful API or WebSocket

---

## Features

* **Training**: Supports custom image datasets (classification, detection, segmentation).
* **Evaluation**: Metrics like accuracy, precision, recall, mAP.
* **Deployment**: One-line deployment with `cerebrium deploy`, auto‐scaling GPU/CPU.
* **Inference**: Real-time endpoints for REST and WebSocket.
* **Configurable**: `cerebrium.toml` lets you adjust hardware, replicas, scaling.
* **Awesome examples**: Integrates best practices from Cerebrium’s example repos.

---

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
1. Add images to custom_test.py and their corresponding labels as two seperate lists. Then run
python test_server.py --customtest

2. Directly just use image:
python test_server.py --image 'path/to/image'

3. Check if cerebrium's api is working:
python test_server.py --health 'path/to/image'
## 🧑‍💻 License

MIT License — see [LICENSE](LICENSE)

--
