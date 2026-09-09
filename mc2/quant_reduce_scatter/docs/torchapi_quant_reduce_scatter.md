# quant_reduce_scatter

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT</term>：支持
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

**说明：** 使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本，否则将会引发报错，比如BUS ERROR等。

## 功能说明

- **接口功能**：实现quant + reduceScatter融合计算。

- **计算公式**：

    $$
    output=Reduce(AllToAllScales * AllToAllData)
    $$

    $$
    AllToAllData=AllToAll(x)
    $$

    $$
    AllToAllScales=AllToAll(scales)
    $$

    其中的Reduce计算是将来自不同rank的数据进行reduce计算。

    - 情形1：如果x数据类型为float8_e4m3fn、float8_e5m2、int8和hifloat8时，入参scale为float32时，先进行reduce_scatter通信，然后进行pertoken-pergroup反量化，得到bfloat16、float16或者float32数据类型的输出。

    - 情形2：如果x数据类型为float8_e4m3fn、float8_e5m2时，入参scales为float8_e8m0时，先进行reduce_scatter通信，然后进行MX反量化，得到bfloat16、float16或者float32数据类型的输出。

## 函数原型

```python
cann_ops_transformer.ops.quant_reduce_scatter.quant_reduce_scatter(
    x: torch.Tensor,
    scales: torch.Tensor,
    hcom: str,
    world_size: int,
    *,
    reduce_op: Optional[str] = 'sum',
    output_dtype: Optional[int] = None,
    x_dtype: Optional[int] = None,
    scales_dtype: Optional[int] = None,
) -> torch.Tensor
```

## 参数说明

### quant_reduce_scatter

- x（Tensor）：必选参数，代表需要进行通信的数据，数据类型支持float8_e4m3fn、float8_e5m2、int8和hifloat8，数据格式支持ND，输入shape支持2维和3维，形如(bs, h)和(b, s, h)，且h满足128对齐。该参数不支持空tensor。

- scales（Tensor）：必选参数，代表通信数据的量化系数，数据类型支持float32、float8_e8m0，数据格式支持ND，当数据类型为float32，量化模式为pertoken-pergroup量化（KG量化），若x为2维(bs, h)，则scales为2维(bs, h/128)，若x为3维(b, s, h)，则scales为3维(b, s, h/128)。当数据类型为float8_e8m0，且x数据类型为float8_e4m3fn，float8_e5m2时，量化模式为MX量化，若x为2维，scales的shape则为3维(bs, h/64, 2)，若x为3维，scales的shape则为4维(b, s, h/64, 2)。h的范围在[1024, 8192]内，且h满足128对齐。该参数不支持空tensor。

- hcom（str）：必须参数，表示通信域handle名。通过get_hccl_comm_name接口获取。

- world_size（int）：必选参数，表示通信域内rank总数，支持2卡、4卡、8卡。

- *：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- reduce_op（str）：可选参数，代表reduce操作类型，默认值为"sum"。当前版本仅支持"sum"。

- output_dtype（int）：可选参数，代表指定输出参数output的数据类型。当前取值为：bfloat16、float16、float32。默认为bfloat16。

- x_dtype（int）：可选参数，代表指定输入参数x的数据类型。当前取值为：float8_e4m3fn、float8_e5m2、int8和hifloat8。默认值为空。

- scales_dtype（int）：可选参数，代表通信数据的量化系数类型。当前取值为：float32、float8_e8m0。默认值为空。

## 返回值说明

### quant_reduce_scatter

- y（Tensor）：表示最终通信结果，数据类型支持float16、bfloat16、float32，shape为(b*s/world_size, h)，数据格式支持ND。

## 约束说明

- 该接口支持训练场景下使用。

- 通信引擎约束：
  - Ascend950DT: 仅支持UB-Memory通信。

- 通信域大小支持2、4、8。

- 通信域使用约束：
  - 同一通信域内仅允许连续执行`aclnnQuantAllReduce`和`aclnnQuantReduceScatter`算子，且该通信域中不允许有其他通信算子。
  - `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。要求满足`HCCL_BUFFSIZE`>= 2 * (`xDataSize` + `scalesDataSize + 1`)。其中`xDataSize`为输入`x`的数据大小，计算公式为：`xDataSize = BS * H * 1 (Byte)`，`scalesDataSize`为`scales`的数据大小，当量化方式为pertoken-pergroup量化时，计算公式为：`scalesDataSize = BS * H / 128 * 4 (Byte)`，当量化方式为mx量化时，计算公式为：`scalesDataSize = BS * H / 32 * 1 (Byte)`。

- 数据类型约束：

    <table style="undefined;table-layout: fixed; width: 1024px"><colgroup>
    <col style="width: 128px">
    <col style="width: 256px">
    <col style="width: 128px">
    <col style="width: 256px">
    </colgroup>
    <thead>
    <tr>
        <th>场景</th>
        <th>x</th>
        <th>scales</th>
        <th>Y</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>pertoken-pergroup量化</td>
        <td>hifloat8、int8、float8_e4m3fn、float8_e5m2</td>
        <td>float32</td>
        <td>float16、bfloat16、float32</td>
    </tr>
    <tr>
        <td>MX量化</td>
        <td>float8_e4m3fn、float8_e5m2</td>
        <td>float8_e8m0</td>
        <td>float16、bfloat16、float32</td>
    </tr>
    </tbody></table>

- shape约束：

    <table style="undefined;table-layout: fixed; width: 1024px"><colgroup>
    <col style="width: 128px">
    <col style="width: 256px">
    <col style="width: 256px">
    </colgroup>
    <thead>
    <tr>
        <th>场景</th>
        <th>x</th>
        <th>scales</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>pertoken-pergroup量化</td>
        <td>2维tensor，shape为(bs, h)</td>
        <td>2维tensor，shape为(bs, h/128)</td>
    </tr>
    <tr>
        <td>pertoken-pergroup量化</td>
        <td>3维tensor，shape为(b, s, h)</td>
        <td>3维tensor，shape为(b, s, h/128)</td>
    </tr>
    <tr>
        <td>MX量化</td>
        <td>2维tensor，shape为(bs, h)</td>
        <td>3维tensor，shape为(bs, h/64, 2)</td>
    </tr>
    <tr>
        <td>MX量化</td>
        <td>3维tensor，shape为(b, s, h)</td>
        <td>4维tensor，shape为(b, s, h/64, 2)</td>
    </tr>
    </tbody></table>

  - H范围仅支持[1024, 8192]，要求128对齐。

## 确定性计算

- 默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  <!-- npu="950" id7 -->
  - **Ascend 950DT**：

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import numpy as np
    from en_dtypes import float8_e8m0

    def run_quant_reduce_scatter(rank, world_size, master_ip, master_port, x_shape, scales_shape, x_dtype, quant_type):
        torch_npu.npu.set_device(rank)
        init_method = "tcp://" + master_ip + ":" + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > "2.0.1":
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
        x_ = torch.randn(x_shape, dtype=torch.float32).to(x_dtype).npu()
        scales_ = torch.rand(size=scales_shape, dtype=torch.float32).npu()
        output = cann_ops_transformer.ops.quant_reduce_scatter.quant_reduce_scatter(
            x_,
            scales_,
            hcom_info,
            world_size,
            scales_dtype=293 if quant_type == "mx" else None
        )
        print("output: ", output)

    if __name__ == "__main__":
        world_size = 2
        master_ip = "127.0.0.1"
        master_port = "50001"
        quant_type = "kg"
        x_shape = [8, 5120]
        scales_shape = [8, 40]
        x_dtype = torch.int8
        mp.spawn(
            run_quant_reduce_scatter,
            args=(world_size, master_ip, master_port, x_shape, scales_shape, x_dtype, quant_type),
            nprocs=world_size,
        )
    ```
  <!-- end id7 -->

- 图模式调用：

  <!-- npu="950" id8 -->
  - **Ascend 950DT**：

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig

    import torch.distributed as dist
    import torch.multiprocessing as mp
    import numpy as np

    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    # npu_backend = tng.get_npu_backend(compiler_config=config)
    npu_backend = tng.get_npu_backend(compiler_config=None)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, scales, hcom, world_size, reduce_op="sum", output_dtype=None, x_dtype=None, scales_dtype=None):
            return cann_ops_transformer.ops.quant_reduce_scatter.quant_reduce_scatter(
                x=x,
                scales=scales,
                hcom=hcom,
                world_size=world_size,
                reduce_op=reduce_op,
                output_dtype=output_dtype,
                x_dtype=x_dtype,
                scales_dtype=scales_dtype
            )

    def run_quant_reduce_scatter_graph(rank, world_size, master_ip, master_port, x_shape, scales_shape, x_dtype):
        torch_npu.npu.set_device(rank)
        init_method = "tcp://" + master_ip + ":" + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > "2.0.1":
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
        x_ = torch.randn(x_shape, dtype=torch.float32).to(x_dtype).npu()
        scales_ = torch.rand(size=scales_shape, dtype=torch.float32).npu()

        cpu_model = Model()
        model = torch.compile(cpu_model, backend=npu_backend, dynamic=False, fullgraph=True)
        output = model(
            x=x_,
            scales=scales_,
            hcom=hcom_info,
            world_size=world_size,
            reduce_op="sum",
            output_dtype=None,
            x_dtype=None,
            scales_dtype=None
        )
        print("output: ", output)

    if __name__ == "__main__":
        world_size = 2
        master_ip = "127.0.0.1"
        master_port = "50001"
        x_shape = [8, 1024]
        scales_shape = [8, 8]
        x_dtype = torch.int8
        mp.spawn(
            run_quant_reduce_scatter_graph,
            args=(world_size, master_ip, master_port, x_shape, scales_shape, x_dtype),
            nprocs=world_size,
        )
    ```
  <!-- end id8 -->
