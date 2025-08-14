FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Install openCV dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install python requirements
RUN pip install --upgrade pip
COPY docker/requirements.txt .
RUN pip install -r requirements.txt
