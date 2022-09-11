FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["/bin/bash", "-c"]
