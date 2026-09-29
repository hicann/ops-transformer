# mamba2_causal_conv1d算子说明

### 功能和实现说明

基于状态空间模型（SSM）的因果卷积，实现MambaV2 Prefill阶段的因果卷积计算。计算流程包含kernel_size=4的depthwise conv1d和SiLU激活。本算子采用纯Vector实现conv1d，并融合bias和SiLU运算以提升性能。

**计算流**

<img src="https://raw.atomgit.com/user-images/assets/7673863/a7e74e1a-1080-4a62-bf45-e172eb790545/image.png" height="300">

### 自定义Kernel输入输出（I/O）

**输入**

| Tensor | shape | dtype |
|-----|-----|-----|
| x   | BDS   | FP32   |
| w   | BDS   | FP32   |
| b   | D     | FP16   |

**输出**

| Tensor | shape | dtype |
|-----|-----|-----|
| out   | BDS   | FP32   |

**参数说明：**

B: batch size
D: dimension
S: sequence len
该算子支持任意长度S

**调用方式**

```
import npu_ops_transformer_ext

out = torch.ops.npu_ops_transformer_ext.mamba2_causal_conv1d(x, w, b)
```

**测试方法**

见当前目录tests/

```
python test_causal_conv1d.py
```
