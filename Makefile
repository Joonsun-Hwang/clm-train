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

SH := /bin/bash
WD := /root/clm-train
CUDA_VERSION := $(shell nvcc --version | grep "release" | cut -c 33-34)

docker-build:
	cp ./docker/Dockerfile-CUDA${CUDA_VERSION} Dockerfile
	docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ./docker

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

test:
	if [ ${CUDA_VERSION} = 10 ]; then \
		echo ${CUDA_VERSION}; \
	else if [ ${CUDA_VERSION} = 11 ]; then \
			echo ${CUDA_VERSION}; \
		fi \
	fi
