## Artifect Evaluation

## Prerequest:

You first need to install CUDA and docker on your machine: https://www.celantur.com/blog/run-cuda-in-docker-on-linux/

## Step 1: Download container

```
docker run --rm --privileged --gpus all -it leeleo3x/hfuse:latest bash
```

## Step 2: Generate Fused Kernels (Optional)

Note that we have provided pre-fused kernels and you can skip this step.

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

- Next, you need to move the fused kernels to the corresponding folders.

```
cd /root
mv ./fused-torch/* ./TorchKernel/fused
mv ./fused-crypto/* ./ethminer/libethash-cuda/
```

## Step 3: Build Fused Kernels



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

- To run DL kernels

```
cd /root/TorchKernel
/usr/local/cuda-11.5/bin/nvprof --csv --log-file performance.csv python3 ./call.py
/usr/local/cuda-11.5/bin/nvprof -f -o dl.nvprof python3 ./call.py
python3 ~/nvprof2json/nvprof2json.py dl.nvprof > dl.json
```

- To run crypto kernels

```
cd /root/ethminer
/usr/local/cuda-11.5/bin/nvprof --csv --log-file performance.csv ./build/fuse/fuser
/usr/local/cuda-11.5/bin/nvprof -f -o crypto.nvprof ./build/fuse/fuser
python3 ~/nvprof2json/nvprof2json.py crypto.nvprof > crypto.json
```

The execution time of each kernel are stored in `/root/ethminer/performance.csv` and `/root/TorchKernel/performance.csv`.

- To visualize kernel execution time results (Figure 7)

```
mv /root/TorchKernel/dl.json /root/HFuse/scripts/data-new
mv /root/ethminer/crypto.json /root/HFuse/scripts/data-new
cd /root/HFuse/scripts/
python3 analyze_nvprof.py ./data-new/dl.json ./data-new/crypto.json
```

The graph is stored in `/root/HFuse/scripts/fused.png`

- To collect kernel metrics (Figure 8)

```
cd /root/ethminer
/usr/local/cuda-11.5/bin/nvprof --csv --log-file metrics.csv --events "elapsed_cycles_pm" --metrics "issue_slot_utilization,achieved_occupancy,stall_memory_dependency" ./build/fuse/fuser
cd /root/TorchKernel
/usr/local/cuda-11.5/bin/nvprof --csv --log-file metrics.csv --events "elapsed_cycles_pm" --metrics "issue_slot_utilization,achieved_occupancy,stall_memory_dependency" python3 call.py
```

You will see the metrics in `/root/ethminer/metrics.csv` and `/root/TorchKernel/metrics.csv` files. Each row shows the name of the 
kenel, the types of the metrics collected, and the value of the metrics.
