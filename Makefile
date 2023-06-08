include .env

ifndef IMAGE_NAME
$(error IMAGE_NAME is not set.)
endif
ifndef IMAGE_TAG
$(error IMAGE_TAG is not set.)
endif
ifndef CONTAINER_NAME
$(error CONTAINER_NAME is not set.)
endif
ifndef CONTAINER_PORT
$(error CONTAINER_PORT is not set.)
endif
ifndef NVIDIA_VISIBLE_DEVICES
$(error NVIDIA_VISIBLE_DEVICES is not set. You can set it like 'all' or '1,2,3')
endif
ifndef PROJECT_NAME
$(error PROJECT_NAME is not set.)
endif

ifndef PYTHON_VERSION
$(error PYTHON_VERSION is not set.)
endif

ifndef CUDA_VERSION
$(error CUDA_VERSION is not set.)
endif
ifndef WHEEL_VERSION
$(error WHEEL_VERSION is not set.)
endif
ifndef CUDNN_VERSION
$(error CUDNN_VERSION is not set.)
endif
ifndef UBUNTU_VERSION
$(error UBUNTU_VERSION is not set.)
endif

SH := /bin/bash
WD := /root/${PROJECT_NAME}
# CUDA_VERSION := $(shell nvcc --version | grep "release" | cut -d ',' -f 2 | cut -c 10-11)

docker-build:
	docker build --progress=plain --no-cache -t ${IMAGE_NAME}:${IMAGE_TAG} $(shell grep -vE '^$$|#' .env | sed 's/^/--build-arg /') ./docker 

docker-run:
	docker run -it -d --restart always -v $(shell pwd):${WD} -p ${CONTAINER_PORT}:${CONTAINER_PORT} --name ${CONTAINER_NAME} --ipc=host --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} ${IMAGE_NAME}:${IMAGE_TAG} ${SH}

docker-exec:
	docker exec -it -w ${WD} ${CONTAINER_NAME} ${SH}

docker-ps-filter:
	docker ps -aq --filter ancestor=${IMAGE_NAME}:${IMAGE_TAG}

docker-start:
	docker start ${CONTAINER_NAME}

docker-stop:
	docker stop ${CONTAINER_NAME}

docker-rm:
	docker rm ${CONTAINER_NAME}

docker-rm-all:
	docker rm $(shell sudo docker ps -aq --filter ancestor=${IMAGE_NAME}:${IMAGE_TAG})

docker-rmi:
	docker rmi ${IMAGE_NAME}:${IMAGE_TAG}

gpustat:
	gpustat -cp -i .1
