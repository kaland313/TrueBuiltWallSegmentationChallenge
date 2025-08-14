image_name=kandras_tbchallenge
container_name=kandras_tbchallenge
workdir=/workspace
host_port=3000
onnx_model_path=$(workdir)/model.onnx

build:
	docker build --file docker/Dockerfile --tag $(image_name)  .

build-serve:
	docker build --file docker/Dockerfile_serve --tag $(image_name)_serve .
	
run: build
	nvidia-docker run \
	-it --rm \
	--shm-size 16G \
	--network host \
	-e NVIDIA_VISIBLE_DEVICES=0,1 \
	-e PYTHONPATH=$(workdir) \
	--name $(container_name) \
	-v $(shell pwd):$(workdir) \
	-v /tmp:/tmp \
	-w $(workdir) \
	$(image_name) \
	/bin/bash

serve-gpu: build-serve
	nvidia-docker run \
	-it --rm \
	--shm-size 16G \
	-e ONNX_MODEL_PATH=$(onnx_model_path) \
	-p $(host_port):3000 \
	--name $(container_name)_serve \
	-v $(shell pwd):$(workdir) \
	-w $(workdir)/src \
	$(image_name)_serve \
	uvicorn api:app --host 0.0.0.0 --port 3000

serve-cpu: build-serve
	docker run \
	-it --rm \
	--shm-size 16G \
	-e ONNX_MODEL_PATH=$(onnx_model_path) \
	-p $(host_port):3000 \
	--name $(container_name)_serve \
	-v $(shell pwd):$(workdir) \
	-w $(workdir)/src \
	$(image_name)_serve \
	uvicorn api:app --host 0.0.0.0 --port 3000
	
stop:
	docker stop $(container_name)

stop-serve:
	docker stop $(container_name)_serve