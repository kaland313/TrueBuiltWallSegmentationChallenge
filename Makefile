image_name=kandras_tbchallenge
container_name=kandras_tbchallenge
workdir_base=/workspace
workdir=$(workdir_base)/tbchallenge
server_port=3000
host_port=3000

run: build
	nvidia-docker run \
	-it --rm \
	--shm-size 16G \
	--network host \
	-e NVIDIA_VISIBLE_DEVICES=0,1 \
	-e PYTHONPATH=$(workdir_base) \
	--name $(container_name) \
	-v $(shell pwd):$(workdir) \
	-v /tmp:/tmp \
	-w $(workdir) \
	$(image_name) \
	/bin/bash

build:
	docker build --tag $(image_name)  .

serve_cpu: build
	docker run \
	-it --rm \
	--shm-size 16G \
	--network host \
	-e PYTHONPATH=$(workdir_base) \
	--name $(container_name)_serve \
	-v $(shell pwd):$(workdir) \
	-v /tmp:/tmp \
	-w $(workdir) \
	$(image_name) \
	/bin/bash

serve: #build
	nvidia-docker run \
	-it --rm \
	--shm-size 16G \
	-e NVIDIA_VISIBLE_DEVICES=0,1 \
	-e PYTHONPATH=$(workdir_base) \
	-p $(host_port):$(server_port) \
	--name $(container_name)_serve2 \
	-v $(shell pwd):$(workdir) \
	-v /tmp:/tmp \
	-w $(workdir) \
	$(image_name) \
	/bin/bash
