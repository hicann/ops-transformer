# FlashAttn 实验版 metadata

## 功能说明

`flash_attn_exp_metadata` 为实验算子 `flash_attn_exp` 计算负载均衡 metadata。实验目录中的实现和 ACLNN 参数与主线 `flash_attn_metadata` 保持一致；`flash_attn_exp` 的 Torch 扩展在调用 metadata 时传入 `head_dim_v=head_dim`、`is_grad_enabled=False`，对应当前支持的 64 维前向场景。

直接使用 Torch 扩展时，先调用 `flash_attn_exp_metadata`，再将返回的 metadata 传入 `flash_attn_exp`；metadata 生成时的 batch、序列长度、head 数、布局和掩码参数必须与主算子一致。参数、返回值和直接调用示例见 [flash_attn_exp 说明](../flash_attn_exp/README.md)。

FA4 兼容接口 `flash_attn_func` 会自动生成 metadata，其参数、限制和调用示例见 [FA4 兼容接口说明](../flash_attn_exp/docs/torchapi_fa4.md)。

## 编译与安装

在仓库根目录加载 CANN 环境后，同时构建实验主算子与 metadata：

```bash
source ${CANN_DIR}/set_env.sh
bash build.sh --pkg --ops=flash_attn_exp,flash_attn_exp_metadata --experimental --soc=ascend950
./build_out/cann-ops-transformer-custom_linux-x86_64.run --install-path=${INSTALL_DIR}
source ${INSTALL_DIR}/vendors/custom_transformer/bin/set_env.bash
```

`CANN_DIR` 和 `INSTALL_DIR` 分别为 CANN 环境和自定义算子的实际安装目录。
