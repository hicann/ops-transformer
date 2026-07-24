
## TyphoonMLA

TyphoonMLA is a mixed naive-absorb MLA kernel for shared prefix. For more technical details on TyphoonMLA, check out our preprint [paper](arxiv.org/abs/2509.21081).

### Folder structure

```
typhoon_mla/
├── src/                    # Source code
│   ├── python_extension/   # Python bindings
│   ├── shared_lib/         # CATLASS kernel impl.
│   └── typhoon_mla.py/     # Python wrappers
│
├── tests/                  # Unit tests
│
├── docs/                   # Documentation
│
├── build/                  # Build output (ignored in VCS)
│
├── bench.py                # Perf. benchmark
├── example.py              # Example usage
├── setup.py                # Setup for python package
├── README.md
└── .gitignore
```

### Requirements

* CATLASS v1.0.0
* CANN toolkit 
* CANN-NNAL (required for TorchNPU absorb baseline)
* Torch & TorchNPU

### Build & compile

1. Clone CATLASS

```
git clone https://gitee.com/ascend/catlass.git
cd catlass 
git checkout v1.0.0
export CATLASS_DIR=$(pwd)
cd ..
```

2. Set CANN environment

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/driver/bin/setenv.bash 
source /usr/local/Ascend/nnal/atb/set_env.sh # Required for the torch_npu absorb baseline
```

3. Compile kernel and python extension

```
cd src
bash install.sh
cd ..
```

### Run TyphoonMLA kernel

```
python example.py 
```

### Verify functional correctness

```
pytest tests
```

### Performance benchmark

```
python bench.py
```

Following benchmark results are obtained in an Ascend 910B2 NPU:

```
batch: 128    shared_seqlen: 4096   nonshared_seqlen: 256    headnum: 64     |  TyphoonMLA:  88.92 ktoken/s  TorchNPU-Absorb:  79.65 ktoken/s  Speedup: 1.12x
batch: 128    shared_seqlen: 4096   nonshared_seqlen: 1024   headnum: 64     |  TyphoonMLA:  89.68 ktoken/s  TorchNPU-Absorb:  72.04 ktoken/s  Speedup: 1.24x
batch: 128    shared_seqlen: 4096   nonshared_seqlen: 4096   headnum: 64     |  TyphoonMLA:  67.53 ktoken/s  TorchNPU-Absorb:  51.06 ktoken/s  Speedup: 1.32x
batch: 128    shared_seqlen: 8192   nonshared_seqlen: 256    headnum: 64     |  TyphoonMLA:  69.64 ktoken/s  TorchNPU-Absorb:  49.66 ktoken/s  Speedup: 1.40x
batch: 128    shared_seqlen: 8192   nonshared_seqlen: 1024   headnum: 64     |  TyphoonMLA:  69.20 ktoken/s  TorchNPU-Absorb:  46.27 ktoken/s  Speedup: 1.50x
batch: 128    shared_seqlen: 8192   nonshared_seqlen: 4096   headnum: 64     |  TyphoonMLA:  56.06 ktoken/s  TorchNPU-Absorb:  36.76 ktoken/s  Speedup: 1.53x
batch: 128    shared_seqlen: 16384  nonshared_seqlen: 256    headnum: 64     |  TyphoonMLA:  51.98 ktoken/s  TorchNPU-Absorb:  28.33 ktoken/s  Speedup: 1.83x
batch: 128    shared_seqlen: 16384  nonshared_seqlen: 1024   headnum: 64     |  TyphoonMLA:  51.99 ktoken/s  TorchNPU-Absorb:  27.22 ktoken/s  Speedup: 1.91x
batch: 128    shared_seqlen: 16384  nonshared_seqlen: 4096   headnum: 64     |  TyphoonMLA:  44.24 ktoken/s  TorchNPU-Absorb:  23.58 ktoken/s  Speedup: 1.88x
batch: 256    shared_seqlen: 4096   nonshared_seqlen: 256    headnum: 64     |  TyphoonMLA: 156.97 ktoken/s  TorchNPU-Absorb: 100.34 ktoken/s  Speedup: 1.56x
batch: 256    shared_seqlen: 4096   nonshared_seqlen: 1024   headnum: 64     |  TyphoonMLA: 142.99 ktoken/s  TorchNPU-Absorb:  88.74 ktoken/s  Speedup: 1.61x
batch: 256    shared_seqlen: 4096   nonshared_seqlen: 4096   headnum: 64     |  TyphoonMLA:  90.73 ktoken/s  TorchNPU-Absorb:  60.07 ktoken/s  Speedup: 1.51x
batch: 256    shared_seqlen: 8192   nonshared_seqlen: 256    headnum: 64     |  TyphoonMLA: 122.38 ktoken/s  TorchNPU-Absorb:  58.57 ktoken/s  Speedup: 2.09x
batch: 256    shared_seqlen: 8192   nonshared_seqlen: 1024   headnum: 64     |  TyphoonMLA: 115.29 ktoken/s  TorchNPU-Absorb:  54.22 ktoken/s  Speedup: 2.13x
batch: 256    shared_seqlen: 8192   nonshared_seqlen: 4096   headnum: 64     |  TyphoonMLA:  78.22 ktoken/s  TorchNPU-Absorb:  42.13 ktoken/s  Speedup: 1.86x
batch: 256    shared_seqlen: 16384  nonshared_seqlen: 256    headnum: 64     |  TyphoonMLA:  88.50 ktoken/s  TorchNPU-Absorb:  31.97 ktoken/s  Speedup: 2.77x
batch: 256    shared_seqlen: 16384  nonshared_seqlen: 1024   headnum: 64     |  TyphoonMLA:  84.80 ktoken/s  TorchNPU-Absorb:  30.53 ktoken/s  Speedup: 2.78x
batch: 256    shared_seqlen: 16384  nonshared_seqlen: 4096   headnum: 64     |  TyphoonMLA:  63.09 ktoken/s  TorchNPU-Absorb:  26.36 ktoken/s  Speedup: 2.39x
batch: 512    shared_seqlen: 4096   nonshared_seqlen: 256    headnum: 64     |  TyphoonMLA: 240.22 ktoken/s  TorchNPU-Absorb: 114.15 ktoken/s  Speedup: 2.10x
batch: 512    shared_seqlen: 4096   nonshared_seqlen: 1024   headnum: 64     |  TyphoonMLA: 185.49 ktoken/s  TorchNPU-Absorb:  99.22 ktoken/s  Speedup: 1.87x
batch: 512    shared_seqlen: 4096   nonshared_seqlen: 4096   headnum: 64     |  TyphoonMLA: 107.00 ktoken/s  TorchNPU-Absorb:  65.05 ktoken/s  Speedup: 1.64x
batch: 512    shared_seqlen: 8192   nonshared_seqlen: 256    headnum: 64     |  TyphoonMLA: 167.67 ktoken/s  TorchNPU-Absorb:  63.27 ktoken/s  Speedup: 2.65x
batch: 512    shared_seqlen: 8192   nonshared_seqlen: 1024   headnum: 64     |  TyphoonMLA: 140.82 ktoken/s  TorchNPU-Absorb:  58.20 ktoken/s  Speedup: 2.42x
batch: 512    shared_seqlen: 8192   nonshared_seqlen: 4096   headnum: 64     |  TyphoonMLA:  89.50 ktoken/s  TorchNPU-Absorb:  44.46 ktoken/s  Speedup: 2.01x
batch: 512    shared_seqlen: 16384  nonshared_seqlen: 256    headnum: 64     |  TyphoonMLA: 110.08 ktoken/s  TorchNPU-Absorb:  33.35 ktoken/s  Speedup: 3.30x
batch: 512    shared_seqlen: 16384  nonshared_seqlen: 1024   headnum: 64     |  TyphoonMLA:  97.08 ktoken/s  TorchNPU-Absorb:  32.00 ktoken/s  Speedup: 3.03x
batch: 512    shared_seqlen: 16384  nonshared_seqlen: 4096   headnum: 64     |  TyphoonMLA:  69.87 ktoken/s  TorchNPU-Absorb:  27.43 ktoken/s  Speedup: 2.55x
batch: 128    shared_seqlen: 4096   nonshared_seqlen: 256    headnum: 128    |  TyphoonMLA:  73.13 ktoken/s  TorchNPU-Absorb:  63.35 ktoken/s  Speedup: 1.15x
batch: 128    shared_seqlen: 4096   nonshared_seqlen: 1024   headnum: 128    |  TyphoonMLA:  66.77 ktoken/s  TorchNPU-Absorb:  56.33 ktoken/s  Speedup: 1.19x
batch: 128    shared_seqlen: 4096   nonshared_seqlen: 4096   headnum: 128    |  TyphoonMLA:  51.37 ktoken/s  TorchNPU-Absorb:  38.40 ktoken/s  Speedup: 1.34x
batch: 128    shared_seqlen: 8192   nonshared_seqlen: 256    headnum: 128    |  TyphoonMLA:  49.56 ktoken/s  TorchNPU-Absorb:  37.40 ktoken/s  Speedup: 1.33x
batch: 128    shared_seqlen: 8192   nonshared_seqlen: 1024   headnum: 128    |  TyphoonMLA:  47.11 ktoken/s  TorchNPU-Absorb:  34.87 ktoken/s  Speedup: 1.35x
batch: 128    shared_seqlen: 8192   nonshared_seqlen: 4096   headnum: 128    |  TyphoonMLA:  38.81 ktoken/s  TorchNPU-Absorb:  26.98 ktoken/s  Speedup: 1.44x
batch: 128    shared_seqlen: 16384  nonshared_seqlen: 256    headnum: 128    |  TyphoonMLA:  32.64 ktoken/s  TorchNPU-Absorb:  20.53 ktoken/s  Speedup: 1.59x
batch: 128    shared_seqlen: 16384  nonshared_seqlen: 1024   headnum: 128    |  TyphoonMLA:  31.33 ktoken/s  TorchNPU-Absorb:  19.71 ktoken/s  Speedup: 1.59x
batch: 128    shared_seqlen: 16384  nonshared_seqlen: 4096   headnum: 128    |  TyphoonMLA:  27.74 ktoken/s  TorchNPU-Absorb:  16.98 ktoken/s  Speedup: 1.63x
batch: 256    shared_seqlen: 4096   nonshared_seqlen: 256    headnum: 128    |  TyphoonMLA: 108.18 ktoken/s  TorchNPU-Absorb:  73.73 ktoken/s  Speedup: 1.47x
batch: 256    shared_seqlen: 4096   nonshared_seqlen: 1024   headnum: 128    |  TyphoonMLA:  89.48 ktoken/s  TorchNPU-Absorb:  64.52 ktoken/s  Speedup: 1.39x
batch: 256    shared_seqlen: 4096   nonshared_seqlen: 4096   headnum: 128    |  TyphoonMLA:  65.29 ktoken/s  TorchNPU-Absorb:  42.24 ktoken/s  Speedup: 1.55x
batch: 256    shared_seqlen: 8192   nonshared_seqlen: 256    headnum: 128    |  TyphoonMLA:  77.19 ktoken/s  TorchNPU-Absorb:  41.07 ktoken/s  Speedup: 1.88x
batch: 256    shared_seqlen: 8192   nonshared_seqlen: 1024   headnum: 128    |  TyphoonMLA:  66.48 ktoken/s  TorchNPU-Absorb:  37.92 ktoken/s  Speedup: 1.75x
batch: 256    shared_seqlen: 8192   nonshared_seqlen: 4096   headnum: 128    |  TyphoonMLA:  53.02 ktoken/s  TorchNPU-Absorb:  29.13 ktoken/s  Speedup: 1.82x
batch: 256    shared_seqlen: 16384  nonshared_seqlen: 256    headnum: 128    |  TyphoonMLA:  51.08 ktoken/s  TorchNPU-Absorb:  21.84 ktoken/s  Speedup: 2.34x
batch: 256    shared_seqlen: 16384  nonshared_seqlen: 1024   headnum: 128    |  TyphoonMLA:  45.94 ktoken/s  TorchNPU-Absorb:  20.98 ktoken/s  Speedup: 2.19x
batch: 256    shared_seqlen: 16384  nonshared_seqlen: 4096   headnum: 128    |  TyphoonMLA:  38.19 ktoken/s  TorchNPU-Absorb:  17.99 ktoken/s  Speedup: 2.12x
batch: 512    shared_seqlen: 4096   nonshared_seqlen: 256    headnum: 128    |  TyphoonMLA: 132.24 ktoken/s  TorchNPU-Absorb:  78.49 ktoken/s  Speedup: 1.68x
batch: 512    shared_seqlen: 4096   nonshared_seqlen: 1024   headnum: 128    |  TyphoonMLA: 106.14 ktoken/s  TorchNPU-Absorb:  70.16 ktoken/s  Speedup: 1.51x
batch: 512    shared_seqlen: 4096   nonshared_seqlen: 4096   headnum: 128    |  TyphoonMLA:  74.39 ktoken/s  TorchNPU-Absorb:  45.39 ktoken/s  Speedup: 1.64x
batch: 512    shared_seqlen: 8192   nonshared_seqlen: 256    headnum: 128    |  TyphoonMLA:  88.66 ktoken/s  TorchNPU-Absorb:  42.26 ktoken/s  Speedup: 2.10x
batch: 512    shared_seqlen: 8192   nonshared_seqlen: 1024   headnum: 128    |  TyphoonMLA:  77.21 ktoken/s  TorchNPU-Absorb:  40.57 ktoken/s  Speedup: 1.90x
batch: 512    shared_seqlen: 8192   nonshared_seqlen: 4096   headnum: 128    |  TyphoonMLA:  58.56 ktoken/s  TorchNPU-Absorb:  30.81 ktoken/s  Speedup: 1.90x
batch: 512    shared_seqlen: 16384  nonshared_seqlen: 256    headnum: 128    |  TyphoonMLA:  56.62 ktoken/s  TorchNPU-Absorb:  22.03 ktoken/s  Speedup: 2.57x
batch: 512    shared_seqlen: 16384  nonshared_seqlen: 1024   headnum: 128    |  TyphoonMLA:  52.21 ktoken/s  TorchNPU-Absorb:  21.97 ktoken/s  Speedup: 2.38x
batch: 512    shared_seqlen: 16384  nonshared_seqlen: 4096   headnum: 128    |  TyphoonMLA:  42.74 ktoken/s  TorchNPU-Absorb:  18.78 ktoken/s  Speedup: 2.28x
```

### Tested on

```
Ascend 910B2
driver: 25.6.rc1.b010
CANN: 8.2.RC1.alpha002
Python: 3.11.10
torch: 2.6.0
torch_npu: 2.6.0rc1
OS: ubuntu: 22.04
```
