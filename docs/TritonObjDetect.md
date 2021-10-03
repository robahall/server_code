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

