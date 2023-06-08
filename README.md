## Fine-tuning Huggingface's CLMs with Accelerate & PEFT


### 1. Installation

#### 1.1. Virtual Environment
##### 1.1.1. Local (Not recommended)
```
conda create -n accelerate
conda activate accelerate

pip install -r requirements.txt
```

##### 1.1.2. TPU
The [torch_xla](https://pytorch.org/xla/release/1.12/index.html) library is used, and the library is pre-installed when you create a TPU VM.  
However, due to version conflicts and bugs, it does not work in versions higher than `tpu-vm-pt-1.10` among the TPU versions.  
After completing TPU creation and configuration through the following shell script.  
```
gcloud alpha compute tpus tpu-vm create %TPU_NAME% --zone=%ZONE% --version=tpu-vm-pt-1.10 --accelerator-type=%TYPE%
gcloud alpha compute tpus tpu-vm ssh %TPU_NAME% --zone %ZONE% --project %PROJECT_NAME%

echo 'export PATH=$HOME/bin:/usr/local/bin:$HOME/.local/bin:$PATH' >> ~/.bashrc
echo 'export XRT_TPU_CONFIG="localservice;0;localhost:51011"' >> ~/.bashrc
source ~/.bashrc

python3 -m pip install --upgrade pip

pip install -r requirements.txt
```

##### 1.1.3. Docker
###### Build container
```
cp env.tmp .env
vi .env
make docker-build
make docker-run
```

###### Execute container (in dev)
```
make docker-start
make docker-exec
```

###### Remove container
```
make docker-stop
make docker-rm
```


#### 1.2 Configuration
Please refer to the following document to set virtual environment up according to your resources.  
https://huggingface.co/docs/accelerate/v0.12.0/en/package_reference/cli#accelerate-config  

There example files of default, DeepSpeed, and TPU configuration are in `accelerate` directory.  
```
accelerate config
```

---


### 2. Input Data
You must have `train.jsonl`, `val.jsonl`, and `test.jsonl` files in the `data` directory, respectively, and this can be set with `--data_dir` among the argument options.  
It is recommended to use all data after preprocessing it in advance, and not recommended to modify the mini-batch after loading it through the DataLoader class.  

The format of `*.jsonl` files are like this for CausalDataset:
```
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
...
```

The format of `*.jsonl` files are like this for InstructDataset:
```
{"instruction": "<intruction text>", "input": "<additional information for instruction>", "output": "<ideal response for instruction>"}
{"instruction": "<intruction text>", "input": "<additional information for instruction>", "output": "<ideal response for instruction>"}
{"instruction": "<intruction text>", "input": "<additional information for instruction>", "output": "<ideal response for instruction>"}=
...
```

If you want to add other special tokens, you could do it at `main()` function in `train.py` file.  

---


### 3. Execution

#### 3.1. Train
##### 3.1.1. Multi-GPU Data Parallel
###### accelerate launch
```
accelerate launch --config_file accelerate_config/default_config.yaml train.py %CHECKPOINT%
```
###### torch distributed launch
```
torchrun train.py %CHECKPOINT%
```

##### 3.1.2. DeepSpeed
To use DeepSpeed, you need yaml and json configuration files. That is, you need to configure DeepSpeed via `accelerate config` command like [1.1.2. Docker](#####112-Docker).  
Please check the following documents:  
https://huggingface.co/docs/accelerate/v0.12.0/en/usage_guides/deepspeed#deepspeed  

There example files of default, DeepSpeed, and TPU configuration are in `accelerate` directory.  
###### accelerate launch
```
accelerate launch --config_file accelerate_config/deepspeed_config.yaml train.py %CHECKPOINT%
```

##### 3.1.3. TPU
###### accelerate launch
```
accelerate launch --config_file accelerate_config/tpu_config.yaml train.py %CHECKPOINT%
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


### 4. PEFT
#### 4.1. LoRA
```
accelerate launch --config_file accelerate_config/default_config.yaml train.py %CHECKPOINT% --use_lora
```

---


### 5. TODO
- Integrate other PEFT methods
