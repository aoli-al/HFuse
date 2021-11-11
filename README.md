## Artifect Evaluation

## Prerequest:

You first need to install CUDA and docker on your machine: https://www.celantur.com/blog/run-cuda-in-docker-on-linux/

## Step 1: Download container

```
docker
```

## Step 2: Generate Fused Kernels

For ML kernels:
```
cd /root
mkdir fused-torch
cd fused-torch
../HFuse/build/tools/llvm-smart-fuser/llvm-smart-fuser ../HFuse/configs/ml_fusion.yaml ../HFuse/configs/ml_kernels.yaml  ../TorchKernel/fused/
```

For crypto kernels:

```
cd /root
mkdir fused-crypto
cd fused-crypto
../HFuse/build/tools/llvm-smart-fuser/llvm-smart-fuser ../HFuse/configs/crypto_fusion.yaml ../HFuse/configs/crypto_kernels.yaml  ../ethminer/libethash-cuda/
```

It will take about 10mins to generate fused kernels.

## Step 3: Run Fused Kernels

- First you need to move the fused kernels to the corresponding folders.

```
cd /root
mv ./fused-torch/* ./TorchKernel/fused
mv ./fused-crypto/* ./ethminer/libethash-cuda/
```


- Build and run PyTorch kernels:

```
cd /root/TorchKernel
./build.sh
```

- Run both fused kernel and original kernel with `nvprof`:

```
```
