# pycuda keccak miner
 A reference pycuda GPU miner for keccak 

## -O3 siricoin (miner-op.py)
 This is a specific optimisation to siricoin, removing all unnecessary functionality but for mining. Overall a >+200% speedup

## Installation
 1: Install Visual Studio (2022|2019|2016) with C++ Desktop Development package
 2: Ensure x64 MSVC is in your PATH (i used C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.32.31326\bin\Hostx64\x64)
 3: Install CUDA Toolkit (https://developer.nvidia.com/cuda-downloads) with Express Settings
 4: Install Python 3.x
 5: Install Python Dependencies with pip (py -m pip install -r requirments.txt on windows, may vary between operation systems)
 6: Edit miner-op and change rewardsRecipient = "0x..." (Line 8) to use your wallet adress
 7: Run miner-op.py

## Benchmark
 I'm Collecting various CPU/GPU benchmarks, contact me on discord if you would like to submit your GPU/Hashrate/Additional Settings (discoflea#3083)

## Hardware
 Cuda capable GPU needed, this should be most modern Nvidia GPUs but you can check here -> (https://developer.nvidia.com/cuda-gpus#compute)
 