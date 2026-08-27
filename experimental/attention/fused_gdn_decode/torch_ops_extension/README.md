# fused_gdn_decode torch_ops_extension

## 功能说明

该目录提供FusedGdnDecode算子的PyTorch扩展接口。安装后可通过如下接口调用：

```python
torch.ops.custom.npu_fused_gdn_decode(
    mixed_qkv,
    a,
    b,
    a_log,
    dt_bias,
    state_ref,
    ssm_state_indices,
    scale,
    softplus_threshold=20.0,
)
```

`state_ref`为原地更新Tensor，输出Tensor shape为`[B, 1, Hv, V]`。

## 编译安装

编译安装前需先安装FusedGdnDecode自定义算子包，并配置自定义算子包环境变量。

```bash
cd experimental/attention/fused_gdn_decode/torch_ops_extension
bash build_and_install.sh
```

安装完成后导入`custom_ops`即可完成`torch.ops.custom`注册：

```python
import torch
import torch_npu
import custom_ops
```
