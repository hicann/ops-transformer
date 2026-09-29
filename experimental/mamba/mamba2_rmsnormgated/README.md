# mamba2_rmsnormgated算子说明

### 功能和实现说明

基于 RMSNorm 和 SiLU 门控的融合算子，实现 MambaV2 Prefill 阶段的 RMSNorm + Gating 计算。计算流程为输入 x 经 SiLU(z) 门控激活后，进行分组 RMSNorm 归一化，再乘以权重 w。

**计算流**

<img src="https://raw.atomgit.com/user-images/assets/7673863/f90c8afa-f740-40c3-a2d4-f470a301b60a/image.png" height="300">

### 自定义Kernel输入输出（I/O）

**输入**

| Tensor | shape | dtype |
|-----|-----|-----|
| x   | BSD   | FP32   |
| w   | D   | FP32   |
| z   | BSD    | FP32   |

**输出**

| Tensor | shape | dtype |
|-----|-----|-----|
| out   | BSD   | FP32   |

**参数说明：**
B: batch size
S: sequence len
D: dimension
额外需要参数
G: ngroups
E: eps

**调用方式**

```
import npu_ops_transformer_ext

out = torch.ops.npu_ops_transformer_ext.mamba2_rmsnormgated(x, z, w, G, E)
```

**测试方法**

见当前目录tests/

```
python test_rmsnormgated.py
```
