# ChunkKdaFwd 调用示例

`test_aclnn_chunk_kda_fwd.cpp`展示通过两段式aclnn API调用`ChunkKdaFwd`的完整流程，包括：

1. 初始化AscendCL、device、context和stream。
2. 构造合法输入、申请device内存并创建连续ND Tensor。
3. 调用`aclnnChunkKdaFwdGetWorkspaceSize`获取workspace大小和executor。
4. 申请workspace，调用`aclnnChunkKdaFwd`执行计算。
5. 将`attn_out`、`Aqk`和`Akk`拷贝至host侧，检查有限值并打印前5个元素。
6. 释放Tensor、device内存和AscendCL资源。

## 示例配置

示例采用最基础的定长 BSND 布局：

| 参数 | 取值或说明 |
| :--- | :--- |
| 产品 | Ascend 950PR&950DT系列产品、Atlas A3系列产品、Atlas A2系列产品 |
| 数据类型 | `q`、`k`、`v`、`attn_out`、`Aqk`、`Akk` 为 BFLOAT16；`g`、`beta`、`a_log` 为 FLOAT32 |
| 规模 | B=1，S=64，H=HV=2，K=V=128，C=64 |
| `layout` | `BSND` |
| gate 配置 | `use_gate_in_kernel=true`、`safe_gate=true`、`lower_bound=-5.0` |
| 可选输入 | `dt_bias`、`initial_state`、`cu_seqlens`、`chunk_indices` 均不传 |
| 可选输出 | 只导出固定的 `attn_out`、`Aqk`、`Akk` |

输入和输出形状如下：

| Tensor | 形状 | 说明 |
| :--- | :--- | :--- |
| `q`、`k` | `[1, 64, 2, 128]` | Query 和 Key |
| `v` | `[1, 64, 2, 128]` | Value |
| `g` | `[1, 64, 2, 128]` | `use_gate_in_kernel=true` 时的 raw gate |
| `beta` | `[1, 64, 2]` | Delta 系数 |
| `a_log` | `[2]` | 每个 Value head 的 gate 参数 |
| `attn_out` | `[1, 64, 2, 128]` | 注意力输出 |
| `Aqk`、`Akk` | `[1, 2, 64, 64]` | chunk 内 Query-Key 系数矩阵和三角求逆结果 |

接口、公式和完整参数约束见 [aclnnChunkKdaFwd](../docs/aclnnChunkKdaFwd.md)。

## 编译与运行

在 ops-transformer 仓库根目录执行。Ascend 950 使用 `--soc=ascend950`；Atlas A2/A3 请替换为构建环境对应的 SoC。

```bash
bash build.sh --pkg --soc=ascend950 --ops=chunk_kda_fwd
bash build_out/cann-ops-transformer-*.run --quiet --install-path=<install_path>
source <install_path>/vendors/custom_transformer/bin/set_env.bash
bash build.sh --run_example chunk_kda_fwd eager cust --soc=ascend950 --vendor_name=custom
```

成功执行后，`build.sh` 会输出 `execute samples success`，示例本身会输出以下内容：

```text
ChunkKdaFwd execution succeeded.
  attnOut[0] = ...
  ...
  Aqk[0] = ...
  ...
  Akk[0] = ...
  ...
```

示例会校验打印结果均为有限值。数值结果可用于确认调用链和输出布局正确；需要与参考实现进行数值比对时，可使用[pytest测试框架](../tests/pytest/README.md)。
