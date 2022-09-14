## Fine-tuning Huggingface's CLMs with Accelerate


### 1. Installation
As of September 2022, most versions are compatible, but when using DeepSpeed's ZERO Optimizer Stage 3, you should use the CUDA-10.2 version.  
If you want to use Stage 3, I recommend using `Dockerfile` as [1.1.2. Docker](#####-1.1.2.-Docker).

#### 1.1. Virtual Environment
##### 1.1.1. Local
```
conda create -n %V_ENV%
conda activate %V_ENV%
pip install -r requirements.txt
accelerate config
```

##### 1.1.2. Docker
```
sudo docker build -t pytorch-cuda10.2:pytorch-cuda10.2 ./
sudo docker run --rm -d -p 12345:12345 --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all pytorch-cuda10.2:pytorch-cuda10.2 /bin/bash
accelerate config
```

#### 1.2 Configuration
Please refer to the following document to set virtual environment up according to your resources.  
https://huggingface.co/docs/accelerate/v0.12.0/en/package_reference/cli#accelerate-config  
```
accelerate config
```

---


### 2. Input Data
You must have `train.jsonl`, `val.jsonl`, and `test.jsonl` files in the `data` directory, respectively, and this can be set with `--data_dir` among the argument options.  
It is recommended to use all data after preprocessing it in advance, and not recommended to modify the mini-batch after loading it through the DataLoader class.  

The format of `*.jsonl` files are like this:
```
{'input_text': 'I am a king.'}
{'input_text': 'I like a game called Lost Ark.'}
...
```

---


### 3. Execution

#### 3.1. Train
##### 3.1.1. Multi-GPU Data Parallel
###### accelerate
```
accelerate launch train.py %CHECKPOINT%
```
###### torch distributed
```
torchrun train.py %CHECKPOINT%
```

##### 3.1.2. Multi-GPU Model Parallel
If you want to train the model in parallel, it is recommended not to use accelerate launch.  
Currently, `accelerate launch` does not provide an argument to control the number of processes, and it is rather slow when parallelizing the model and training with multi-process.  
Among the arguments, it works by acquiring an empty space (space to be loaded with data) of each device through `--extra memory`, and dividing and allocating the model to the rest.  
###### torch distributed
```
torchrun --nproc_per_node 1 train.py %CHECKPOINT% --model_parallel
```

##### 3.1.3. DeepSpeed
To use DeepSpeed, you need yaml and json configuration files. That is, you need to configure DeepSpeed via `accelerate config` command.  
Please check the following documents:  
https://huggingface.co/docs/accelerate/v0.12.0/en/usage_guides/deepspeed#deepspeed
###### accelerate
```
accelerate launch --config_file /absolute/path/to/deepspeed/config/file train.py %CHECKPOINT%
```

##### 3.1.4. TPU
###### TPU setting
The [torch_xla](https://pytorch.org/xla/release/1.12/index.html) library is used, and the library is pre-installed when you create a TPU VM.  
However, due to version conflicts and bugs, it does not work in versions higher than `tpu-vm-pt-1.10` among the tpu versions.  
After completing tpu creation and configuration through the following shell script, follow the instructions in [1.1.1. Local](#####-1.1.1.-Local).  
```
gcloud alpha compute tpus tpu-vm create %TPU_NAME% --zone=%ZONE% --version=tpu-vm-pt-1.10 --accelerator-type=%TYPE%
gcloud alpha compute tpus tpu-vm ssh %TPU_NAME% --zone %ZONE% --project %PROJECT_NAME%

echo 'export PATH=$HOME/bin:/usr/local/bin:$HOME/.local/bin:$PATH' >> ~/.bashrc
echo 'export XRT_TPU_CONFIG="localservice;0;localhost:51011"' >> ~/.bashrc
source ~/.bashrc

python3 -m pip install --upgrade pip
```
###### accelerate
```
accelerate launch train.py %CHECKPOINT%
```

#### 3.2. Test
```
python test.py --saved_model %CHECKPOINT%
```

#### 3.3. Inference
```
python inference.py --saved_model %CHECKPOINT%
```

---


### 4. Fine-Tuning Methods
#### 4.1. Pre-Fine-Tuning
```
accelerate launch train.py %PRE_CHECKPOINT%
accelerate launch train.py %POST_CHECKPOINT% --saved_model %PRE_CHECKPOINT%
```

#### 4.2. Adapter
```
accelerate launch train.py %CHECKPOINT% --add_adpater
python test.py %CHECKPOINT% --add_adapter
```

---


### 5. TODO
- Verify Docker Configuration and ZERO Optimizer Stage 3  
- Occur error when using specific devices through `CUDA_VISIBLE_DEVICES` (like `CUDA_VISIBLE_DEVICES=1,2 accleerate launch train.py %CHECKPOINT% --model_parallel`) with distirbuted type is Multi-GPU model parallel or DeepSpeed  
