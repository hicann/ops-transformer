# RainFusionAttention测试套件

## 概述

本目录包含RainFusionAttention算子的完整测试套件。

## 测试用例

### 1. TND格式测试

- **基础TND格式测试**: 验证标准TND [T,N,D]格式的正确性
- 配置: batch=1, qSeq=256, kvSeq=512, heads=8

### 2. 不同blockShape配置测试

- **小块测试(32x32)**: 验证小块稀疏模式
- **大块测试(128x128)**: 验证大块稀疏模式  
- **非对称块测试(64x128)**: 验证非对称块配置

### 3. 边界条件测试

- **非对齐序列长度**: qSeq=250, kvSeq=500 (不能被64整除)
- **小序列长度**: qSeq=64, kvSeq=128 (极小序列)
- **单头配置**: numHeads=1 (边界情况)
- **多batch**: batch=2 (多batch处理)

### 4. 稀疏度测试

- **高稀疏度**: sparsity=0.1 (10%，极稀疏)
- **低稀疏度**: sparsity=0.8 (80%，接近密集)

## 编译和运行

```bash
# 编译测试
cd /path/to/ops-transformer-dev
bash build.sh --soc=ascend910b

# 运行测试
./test_rain_fusion_attention
```

## 测试输出

测试会输出每个用例的执行结果：

```
--- Test: Basic TND Format ---
Config: batch=1, qSeq=256, kvSeq=512, heads=8, blockShape=[64,64]
Result: PASS

======================================
  Test Summary
======================================
Total tests:   10
Passed:        10 (100%)
Failed:        0
Skipped:       0
======================================
```

## 添加新测试

在`test_rain_fusion_attention.cpp`的`tests`向量中添加新的`TestConfig`:

```cpp
tests.push_back({
    batch,          // batch size
    qSeqlen,        // Q sequence length
    kvSeqlen,       // KV sequence length
    numHeads,       // number of heads
    numKvHeads,     // number of KV heads
    headDim,        // head dimension
    blockShapeX,    // block shape X
    blockShapeY,    // block shape Y
    sparsity,       // sparsity ratio
    "Test Name"     // test name
});
```

## 注意事项

1. 测试需要在有昇腾设备的环境中运行
2. 确保CANN环境已正确配置
3. 测试使用固定随机种子以保证结果可复现
4. 当前仅支持TND格式，BNSD格式测试会被跳过
