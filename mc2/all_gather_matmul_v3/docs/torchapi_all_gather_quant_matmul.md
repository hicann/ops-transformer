# all_gather_quant_matmul

## 产品支持情况

<!-- npu="950" id1 -->

- <term>Ascend 950PR/Ascend 950DT</term>：支持

<!-- end id1 -->

<!-- npu="A3" id2 -->

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持

<!-- end id2 -->

<!-- npu="910b" id3 -->

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持

<!-- end id3 -->

<!-- npu="310b" id4 -->

- <term>Atlas 200I/500 A2 推理产品</term>：不支持

<!-- end id4 -->

<!-- npu="310p" id5 -->

- <term>Atlas 推理系列产品</term>：不支持

<!-- end id5 -->

<!-- npu="910" id6 -->

- <term>Atlas 训练系列产品</term>：不支持

<!-- end id6 -->

## 功能说明

- **接口功能**：

  `cann_ops_transformer.all_gather_quant_matmul`完成AllGather通信与MX量化矩阵乘法的融合计算。在多卡场景下，各卡先通过URMA all-gather收集所有卡的左矩阵及其MX缩放因子，再与本地的右矩阵做MX FP8/FP4量化矩阵乘法，输出BF16或FP16结果。
- **计算公式**：

  本算子当前仅支持MX量化场景：`x1`为`(M, K)`、`x2`为`(N, K)`，`x1_scale`为`(M, ceilDiv(K, 64), 2)`、`x2_scale`为`(N, ceilDiv(K, 64), 2)`。入参`x1`和`x1_scale`进行AllGather后，对`x1`、`x2`进行MatMul计算，然后按K轴分组（group size = 32）进行dequant操作：

  $$
  output=\sum_{0}^{\left \lfloor \frac{K}{blockSize=32} \right \rfloor}(AllGather(x1)@x2*(AllGather(x1\_scale)*x2\_scale)) + bias
  $$

  `bias`为可选累加项，输出形状为`(M * rankSize, N)`，其中 rankSize 为 参与allgather matmul计算的卡数。

## 函数原型

```python
cann_ops_transformer.all_gather_quant_matmul(
    x1,
    x2,
    group,
    *,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    group_sizes=None,
    x1_dtype=None,
    x2_dtype=None,
    x1_scale_dtype=None,
    x2_scale_dtype=None,
    y_dtype=None,
    comm_mode=None,
) -> (Tensor, Tensor, Tensor)
```

## 参数说明

| 参数名             | 参数类型     | 可选/必选 | 描述                                                                                                                                                                                                                                                                                  | 数据类型                                                                                 | 数据格式       | 维度(shape)              |
| ------------------ | ------------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | -------------- | ------------------------ |
| `x1`             | Tensor       | 必选      | 左矩阵，对应各卡本地的`x1[M, K]`。MXFP4场景以uint8打包存储（2个fp4占1字节），实际传入的K轴大小为`K/2`，需配合`x1_dtype=296`（`float4_e2m1`）。                                                                                                                                                   | `float8_e4m3fn`、`float8_e5m2`、`float4_e2m1` | ND             | `(M, K)`               |
| `x2`             | Tensor       | 必选      | 右矩阵，对应公式中X2，需以转置视图（非连续tensor）传入，即先构造`(N, K)`再`.t()`。MXFP4场景以uint8打包存储，实际传入的K轴大小为`K/2`，需配合`x2_dtype=296`（`float4_e2m1`），且`x1`/`x2`必须同为FP4。                                                                                                                                | `float8_e4m3fn`、`float8_e5m2`、`float4_e2m1` | ND             | `(N, K)`               |
| `group`          | ProcessGroup | 必选      | distributed通信域，用于AllGather通信。                                                                                                                                                                                                                                          | -                                                                                        | -              | -                        |
| `bias`           | Tensor       | 可选      | 偏置项，矩阵乘后累加；默认值为`None`，传入`None`时不叠加偏置。                                                                                                                                                                                                                    | `float32`                                                                        | ND             | `(N,)`                 |
| `x1_scale`       | Tensor       | 可选      | `x1`的MX量化scale，默认值为`None`，MX量化场景必传。                                                                                                                                                                                                                               | `float8_e8m0`                                                                 | ND             | `(M, ceil(K / 64), 2)` |
| `x2_scale`       | Tensor       | 可选      | `x2`的MX量化scale，默认值为`None`，MX量化场景必传。                                                                                                                                                                                                                               | `float8_e8m0`                                                                 | ND             | `(N, ceil(K / 64), 2)` |
| `group_sizes`    | List[int]    | 可选      | MX量化group大小，MX场景仅支持`[1, 1, 32]`；默认值为`None`，传入`None`时内部会自动推导，推导规则详见约束。 | int                                                                                      | -              | -                        |
| `x1_dtype`       | int          | 可选      | `x1`的dtype，默认值为`None`，传入`None`时使用输入`x1`的dtype。MXFP8场景传`None`（torch原生支持float8_e4m3fn/float8_e5m2，无需传入）；MXFP4场景传`296`（float4_e2m1）。                                                       | int                                                                                      | -              | -                        |
| `x2_dtype`       | int          | 可选      | `x2`的dtype，传入`None`时使用输入`x2`的dtype。FP8场景传`None`（torch原生支持float8_e4m3fn/float8_e5m2，无需传入）；MXFP4场景传`296`（float4_e2m1）。                                                                           | int                                                                                      | -              | -                        |
| `x1_scale_dtype` | int          | 可选      | `x1_scale`的dtype，传入`None`时使用`x1_scale`的dtype。torch无原生float8_e8m0类型时，传`293`（float8_e8m0）。                                                                                                                                                    | int                                                                                      | -              | -                        |
| `x2_scale_dtype` | int          | 可选      | `x2_scale`的dtype，传入`None`时使用`x2_scale`的dtype。torch无原生float8_e8m0类型时，传`293`（float8_e8m0）。                                                                                                                                                    | int                                                                                      | -              | -                        |
| `y_dtype`        | int          | 可选      | 输出`y`的数据类型，仅支持传`15`（BF16）或`5`（FP16）；传入`None`时默认为BF16。                                                                                                                                                                                                | int                                                                                      | -              | -                        |
| `comm_mode`      | str          | 可选      | 通信模式，当前仅支持`"aiv_urma"`，传入`None`时默认值为`"ai_cpu"`。                                                                                                                                                                                                                                | string                                                                                   | -              | -                        |

## 返回值说明

| 参数名         | 参数类型 | 描述                                                                                   | 数据类型                        | 数据格式       | 维度(shape)           |
| -------------- | -------- | -------------------------------------------------------------------------------------- | ------------------------------- | -------------- | --------------------- |
| `y`          | Tensor   | AllGather与矩阵乘融合计算输出。                                                        | 由`y_dtype`指定（BF16或FP16） | ND             | `(M * rankSize, N)` |
| `gather_out` | Tensor   | AllGather通信中间输出，预留参数，当前版本未实现（返回shape为`(0,)`的空占位tensor）。 | 同`x1`                        | ND             | `(0,)`              |
| `amax_out`   | Tensor   | amax输出，预留参数，当前版本未实现（返回shape为`(0,)`的空占位tensor）。              | `float32`               | ND             | `(0,)`              |

## 约束说明

- 适用场景：该接口支持训练、推理场景下使用。
- 调用方式：该接口支持单算子模式调用，需要在多卡通信域下使用。
- 仅支持Ascend 950系列产品。
- 输入输出Tensor的数据格式仅支持ND。
- 输入`x1`为2维，shape为`(M, K)`；输入`x2`为2维，须传入`[N, K]`存储的`.t()`转置视图。`x1`与`x2`的K轴相等，`K`取值范围为[256, 65535)；FP4场景`K`须为偶数。
- `M`（单卡M轴长度）、`N`取值范围为[1, 2147483647]（INT32_MAX）。
- `bias`为1维，shape为`(N,)`，dtype为`float32`。
- `x1_scale`、`x2_scale`为3维，shape分别为`(M, ceil(K/64), 2)`、`(N, ceil(K/64), 2)`，dtype为`float8_e8m0`（torch原生不支持时，需uint8存储搭配scale_dtype=293）。
- 当前仅支持单算子调用，不支持图模式。
- 输出`y`为2维，shape为`(M * rank_size, N)`，dtype仅支持BF16/FP16。
- `x1`与`x2`需同为FP8（float8_e4m3fn与float8_e5m2可混用）或同为FP4。
- `rank_size`需为2、4、8、16之一。
- `comm_mode`当前仅支持`"aiv_urma"`。
- `group_sizes`在MX场景仅支持`[1, 1, 32]`（显式传入或自动推导结果均须满足）。`group_sizes`可以表示为`[groupSizeM, groupSizeN, groupSizeK]`。当该list中有1个或多个为0，会根据x1/x2/x1Scale/x2Scale输入shape重新设置groupSizeM、groupSizeN、groupSizeK用于计算。设置原理：`x1_scale`/`x2_scale`为3维，shape为`(dim0, ceil(K/64), 2)`，其中第0维（dim0）对应M（或N）轴，第1维（dim1，即`ceil(K/64)`）为K轴分组数，第2维（dim2，即`2`）为每组内scale数。如果groupSizeM=0，表示m方向量化分组值由接口推导，推导公式为`groupSizeM = x1.M / x1_scale.dim0`（需保证x1.M能被x1_scale.dim0整除），其中x1.M与x1 shape中的M一致，x1_scale.dim0为x1Scale shape第0维（其值为M）；如果groupSizeK=0，表示k方向量化分组值由接口推导，推导公式为`groupSizeK = x1.K / (x1_scale.dim1 × x1_scale.dim2)`（需保证x1.K能被`x1_scale.dim1 × x1_scale.dim2`整除），其中x1.K与x1 shape中的K一致，`x1_scale.dim1 × x1_scale.dim2`为x1Scale shape第1维与第2维的乘积（即`ceil(K/64) × 2`，表示k方向scale元素总数）；如果groupSizeN=0，表示n方向量化分组值由接口推导，推导公式为`groupSizeN = x2.N / x2_scale.dim0`（需保证x2.N能被x2_scale.dim0整除），其中x2.N与x2 shape中的N一致，x2_scale.dim0为x2Scale shape第0维（其值为N）。
- 通信域`group`名称长度需在`[1, 128)`范围内。
- 通信buffer大小约束：本算子使用HCCL内置通信buffer完成AllGather通信，该buffer大小由环境变量`HCCL_BUFFSIZE`（单位MB，不配置时默认200MB，传入到框架层时会分配两倍，为400MB）控制。要求`通信数据总量 + 2MB <= HCCL_BUFFSIZE * 2`，其中通信数据总量为`rank_size * M * (K*sizeof(x1_dtype) + ceil(K / 64) * 2) + 2MB`字节，其中`sizeof(x1_dtype)`在FP8场景为1字节、FP4场景为0.5字节。可通过`export HCCL_BUFFSIZE=<取值>`设置所需`HCCL_BUFFSIZE`环境变量。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用示例（MX FP8 场景，4 卡），文件名为demo.py，示例细节如下：

  ```python
  import os
  import math
  from torch.multiprocessing import Process

  os.environ.setdefault("HCCL_BUFFSIZE", "2000")  # 通信 buffer，单位 MB

  import torch
  import torch_npu
  import torch.distributed as dist
  import cann_ops_transformer


  def run_all_gather_quant_matmul(rank, world_size, m, k, n):
      torch_npu.npu.set_device(rank % 4)

      dist.init_process_group(
          backend="hccl",
          rank=rank,
          world_size=world_size,
          init_method="tcp://127.0.0.1:29500",
      )
      group = dist.new_group(ranks=list(range(world_size)))

      # 构造输入（各 rank 生成不同的数据）
      torch.manual_seed(rank)
      x1 = torch.randn(m, k).to(torch.float8_e4m3fn).npu()
      x2 = torch.randn(n, k).to(torch.float8_e4m3fn).npu()
      x1_scale = torch.randint(0, 256, (m, math.ceil(k / 64), 2), dtype=torch.uint8).npu()
      x2_scale = torch.randint(0, 256, (n, math.ceil(k / 64), 2), dtype=torch.uint8).npu()

      # 调用算子（x2 须传入 [N, K] 存储的 .t() 转置视图）
      y, gather_out, amax_out = cann_ops_transformer.all_gather_quant_matmul(
          x1,
          x2.t(),
          group,
          x1_scale=x1_scale,
          x2_scale=x2_scale,
          group_sizes=[1, 1, 32],
          x1_scale_dtype=293,   # fp8_e8m0（uint8 存储，需用 enum 覆盖 dtype）
          x2_scale_dtype=293,
          y_dtype=15,           # BF16
          comm_mode="aiv_urma",
      )
      torch.npu.synchronize()

      print(f"[rank {rank}] y.shape={tuple(y.shape)}, y.dtype={y.dtype}")
      dist.barrier()
      dist.destroy_process_group()


  if __name__ == "__main__":
      world_size = 4
      m, k, n = 2048, 4096, 2560

      torch.multiprocessing.set_start_method("spawn", force=True)

      procs = []
      for rank in range(world_size):
          p = Process(target=run_all_gather_quant_matmul, args=(rank, world_size, m, k, n))
          p.start()
          procs.append(p)

      for p in procs:
          p.join()
  ```
  运行方式：

  ```bash
  # 需在多卡 NPU 环境下执行（4 卡）
  python3 demo.py
  ```
