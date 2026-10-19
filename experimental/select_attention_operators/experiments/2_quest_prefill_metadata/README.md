# Experiment 2 - Quest Prefill Metadata (Prefill)

This experiment checks the functional correctness and the performance of quest prefill metadata kernel that should be launched to create the quest metadata blocks out of the KV cache that had already underwent prefill.

## Scenario parameters

 - `BLOCK_SIZE` - fixed to 128 as commonly used in paged attention
 - `D` - fixed to 128 as commonly used in LLMs
 - `N` - fixed to 8 as in Qwen3-8B
 - `B` - sweep across big batch sizes
 - `MKBPR` - number of KV blocks per request (dictates the sequence length)
 - `MMBPR` - directly derived from _MKBPR_ as: _cdiv(MKBPR, BLOCK_SIZE)_

## Test the operator first

To verify it was properly built

```bash
pytest -k basic -v  # run only basic sweep
pytest .            # run all tests
```

## Run a single scenario

To run an in-depth comparison between a kernel and its reference implementation in python, dump all output tensors:

```bash
python test_quest_prefill_metadata.py
```

## Run benchmarking

To see latency and memory bandwidth

```bash
 python benchmark_quest_prefill_metadata.py
```

when setting the DTYPE to float16, benchmarking on x86 host and 910B4 card shows that the kernel reaches at least 0.55 TB/sec out of 0.80 TB/sec nominal Global memory BW of 910B4 - 69% bandwidth utilization.

```bash
==========================================================================================================
  DTYPE=torch.float16  BLOCK_SIZE=128  D=128  SAME_SEQ_LEN_ALL_REQS=True
==========================================================================================================
  N   B    Seq_len   Outputs_equal Ref_Latency_[usec] Our_Latency_[usec]  Ref_BW_[TB/sec]  Our_BW_[TB/sec]
----------------------------------------------------------------------------------------------------------
  8  10       8064             yes           93027.76             309.27            0.002            0.551
  8  10      10240             yes          117024.04             384.37            0.002            0.559
  8  10      12032             yes          136078.70             484.75            0.002            0.519
  8  10      16384             yes          182469.81             602.44            0.002            0.566
  8  10      19200             yes          219475.81             718.18            0.002            0.562
  8  10      25600             yes          329046.24             948.14            0.002            0.564
  8  10      32768             yes          417126.71            1321.51            0.002            0.516
  8  20       8064             yes          208446.04             664.72            0.002            0.513
  8  20      10240             yes          270382.06             835.40            0.002            0.515
  8  20      12032             yes          314411.11             889.26            0.002            0.566
  8  20      16384             yes          446804.24            1325.83            0.002            0.514
  8  20      19200             yes          492977.95            1572.19            0.002            0.514
  8  20      25600             yes          671196.25            1890.16            0.002            0.566
  8  20      32768             yes          835718.99            2653.32            0.002            0.514
  8  24       8064             yes          258454.59             756.71            0.002            0.540
  8  24      10240             yes          329535.64             943.38            0.002            0.547
  8  24      12032             yes          376479.13            1213.10            0.002            0.498
  8  24      16384             yes          518973.06            1635.08            0.002            0.500
  8  24      19200             yes          609663.98            1785.78            0.002            0.543
  8  24      25600             yes          807009.20            2584.06            0.002            0.497
  8  24      32768             yes         1023406.74            2992.35            0.002            0.547
  8  32       8064             yes          349012.08            1079.05            0.002            0.505
  8  32      10240             yes          407759.97            1233.08            0.002            0.558
  8  32      12032             yes          515659.63            1438.41            0.002            0.560
  8  32      16384             yes          681750.24            2144.14            0.002            0.509
  8  32      19200             yes          809321.04            2323.78            0.002            0.556
  8  32      25600             yes         1036969.16            3398.52            0.002            0.504
  8  32      32768             yes         1353555.26            4333.40            0.002            0.503
==========================================================================================================
```

similar performance for bfloat16:

```bash
==========================================================================================================
  DTYPE=torch.bfloat16  BLOCK_SIZE=128  D=128  SAME_SEQ_LEN_ALL_REQS=True
==========================================================================================================
  N   B    Seq_len   Outputs_equal Ref_Latency_[usec] Our_Latency_[usec]  Ref_BW_[TB/sec]  Our_BW_[TB/sec]
----------------------------------------------------------------------------------------------------------
  8  10       8064             yes           91797.47             341.11            0.002            0.500
  8  10      10240             yes          119940.88             416.60            0.002            0.516
  8  10      12032             yes          142493.83             444.33            0.002            0.566
  8  10      16384             yes          190612.51             659.39            0.002            0.517
  8  10      19200             yes          228080.26             788.18            0.002            0.512
  8  10      25600             yes          331777.49            1042.17            0.002            0.513
  8  10      32768             yes          422492.27            1208.98            0.002            0.564
  8  20       8064             yes          213834.41             610.95            0.002            0.558
  8  20      10240             yes          268580.89             767.21            0.002            0.560
  8  20      12032             yes          320251.59             980.86            0.002            0.513
  8  20      16384             yes          425229.09            1209.94            0.002            0.563
  8  20      19200             yes          501996.54            1439.80            0.002            0.561
  8  20      25600             yes          677030.97            2083.51            0.002            0.513
  8  20      32768             yes          853946.94            2409.99            0.002            0.566
  8  24       8064             yes          259979.37             827.37            0.002            0.494
  8  24      10240             yes          311260.29            1038.10            0.002            0.497
  8  24      12032             yes          370500.20            1109.80            0.002            0.544
  8  24      16384             yes          523368.08            1489.21            0.002            0.549
  8  24      19200             yes          616558.68            1952.58            0.002            0.496
  8  24      25600             yes          808925.46            2347.28            0.002            0.547
  8  24      32768             yes         1033353.92            3306.62            0.002            0.495
  8  32       8064             yes          354466.72             989.11            0.002            0.551
  8  32      10240             yes          443984.86            1354.58            0.002            0.508
  8  32      12032             yes          523036.50            1585.94            0.002            0.508
  8  32      16384             yes          704769.45            1947.90            0.002            0.560
  8  32      19200             yes          805073.97            2558.09            0.002            0.505
  8  32      25600             yes         1086389.57            3075.26            0.002            0.556
  8  32      32768             yes         1375007.49            3912.25            0.002            0.557
==========================================================================================================
```
