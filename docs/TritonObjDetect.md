# Notes on simple  object detection using triton on CPU

```bash
 docker pull nvcr.io/nvidia/tritonserver:21.08-py3
```

##### Download pre-trained model from torchvision
* ssd_mobilenet_v3

##### Create config.pbtxt
https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md

##### Arranged models repo


##### Build gRPC Client
https://github.com/triton-inference-server/client/blob/main/src/python/examples/grpc_image_client.py



##### Run Server 
 ```bash
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/home/rob/PycharmProjects/server_code/models:/models nvcr.io/nvidia/tritonserver:21.08-py3 tritonserver --model-repository=/models
```

Next steps:
#### Run inference Server
Need to install tritonclient server

```bash
pip install tritonclient[all]
```

# Notes on object detection using Triton on GPU

### Need to install NVIDIA Container Toolkit

#### Using Linux Mint based on Ubuntu

```bash
cat /etc/upstream-release/lsb-release 
```

Currently running Mint 20 so ubuntu20.04

```bash
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list`
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

```

TODO: 
* Had to update CUDA 11.5

* Update Pytorch to 11.5?
* Reboot TritonServer. 
* Still failing on forward pass. 