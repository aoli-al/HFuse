## Artifect Evaluation

## Prerequest:

You first need to install CUDA and docker on your machine: https://www.celantur.com/blog/run-cuda-in-docker-on-linux/

## Step 1: Download container

```
docker run -–rm -–privileged -–gpus all -it leeleo3x/hfuse:latest bash
```

## Step 2: Generate Fused Kernels

- Fuse DL kernels:
```
cd /root
mkdir fused-torch
cd fused-torch
HFUSE_PARALLEL=1 ../HFuse/build/tools/llvm-smart-fuser/llvm-smart-fuser ../HFuse/configs/ml_fusion.yaml ../HFuse/configs/ml_kernels.yaml  ../TorchKernel/fused/
```


- Fuse crypto kernels:
```
cd /root
mkdir fused-crypto
cd fused-crypto
HFUSE_PARALLE=0 ../HFuse/build/tools/llvm-smart-fuser/llvm-smart-fuser ../HFuse/configs/crypto_fusion.yaml ../HFuse/configs/crypto_kernels.yaml  ../ethminer/libethash-cuda
```

You may use `HFUSE_PARALLEL` to enable parallel fusing. 
Note that you can only use it while fusing DL kernels. 
Fusing all kernels takes ~30min. 

## Step 3: Build Fused Kernels

- First you need to move the fused kernels to the corresponding folders.

```
cd /root
mv ./fused-torch/* ./TorchKernel/fused
mv ./fused-crypto/* ./ethminer/libethash-cuda/
```


- Build DL kernels:

```
cd /root/TorchKernel
./build.sh
```

- Build crypto kernels:


```
cd /root/ethminer
mkdir build
cd build
cmake ..
make fuser -j4
```

Building two projects take ~30min.

## Step 4: Run Fused Kernels


```
cd /root/TorchKernel
/usr/local/cuda-11.5/bin/nvprof --csv --log-file performance.csv python3 ./call.py
/usr/local/cuda-11.5/bin/nvprof -f -o dl.nvprof python3 ./call.py
```