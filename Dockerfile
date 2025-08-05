# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
# FROM pytorchlightning/pytorch_lightning:2.3.3-py3.10-torch2.0-cuda11.8.0

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# RUN apt-get update && apt-get install -y \
#     git 
# RUN pip install opencv-python pycocotools matplotlib onnxruntime onnx
# RUN pip install git+https://github.com/facebookresearch/segment-anything.git

