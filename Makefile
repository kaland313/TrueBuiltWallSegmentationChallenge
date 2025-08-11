image_name=kandras_tbchallenge
container_name=kandras_tbchallenge
workdir_base=/workspace
workdir=$(workdir_base)/tbchallenge
server_port=3000
host_port=3000
onnx_model_path=$(workdir)/model.onnx

build:
	docker build --tag $(image_name)  .

build-serve:
	docker build --file Dockerfile_serve --tag $(image_name)_serve .
	
run: build
	nvidia-docker run \
	-it --rm \
	--shm-size 16G \
	--network host \
	-e NVIDIA_VISIBLE_DEVICES=0,1 \
	-e PYTHONPATH=$(workdir_base) \
	-e MODEL_PATH=$(model_path) \
	--name $(container_name) \
	-v $(shell pwd):$(workdir) \
	-v /tmp:/tmp \
	-w $(workdir) \
	$(image_name) \
	/bin/bash

serve: build-serve
	nvidia-docker run \
	-it --rm \
	--shm-size 16G \
	-e PYTHONPATH=$(workdir_base) \
	-e ONNX_MODEL_PATH=$(onnx_model_path) \
	-p $(host_port):$(server_port) \
	--name $(container_name)_serve \
	-w $(workdir) \
	$(image_name) \
	/bin/bash
