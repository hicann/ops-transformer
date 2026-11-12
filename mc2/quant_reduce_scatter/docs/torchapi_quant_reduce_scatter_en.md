# quant_reduce_scatter

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:37:57.042Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

**NOTE:** When using this API, ensure that both the driver firmware package and the CANN package are of the matched 9.2.0 version or a later matched version. Otherwise, errors such as BUS ERROR will occur.

## Function

- **Description**: Implements fused quant + reduceScatter computation.

- **Formula**:

    $$
    output=Reduce(AllToAllScales * AllToAllData)
    $$

    $$
    AllToAllData=AllToAll(x)
    $$

    $$
    AllToAllScales=AllToAll(scales)
    $$

    The Reduce computation here performs a reduce operation on data from different ranks.

    - Case 1: If x is of the float8_e4m3fn, float8_e5m2, int8, or hifloat8 data type and the input parameter scales is float32 and has the same number of dimensions as x, reduce_scatter communication is performed first, followed by pertoken-pergroup dequantization. The output is of the bfloat16, float16, or float32 data type.

    - Case 2: If x is of the float8_e4m3fn or float8_e5m2 data type and the input parameter scales is float8_e8m0, reduce_scatter communication is performed first, followed by MX dequantization. The output is of the bfloat16, float16, or float32 data type.

## Function Prototype

```python
cann_ops_transformer.ops.quant_reduce_scatter(
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

## Mandatory

### quant_reduce_scatter

- x (Tensor): Mandatory parameter. Data to be communicated. Data types: float8_e4m3fn, float8_e5m2, int8, and hifloat8. Data format: ND. Input shape: 2D or 3D, in the form of (bs, h) or (b, s, h), where h is 128-aligned. Empty tensors are not supported for this parameter.

- scales (Tensor): Mandatory parameter. Quantization coefficients of the data to be communicated. Data types: float32 and float8_e8m0. Data format: ND. If the data type is float32, the quantization mode is pertoken-pergroup quantization (KG quantization). If x is 2D (bs, h), scales is 2D (bs, h/128). If x is 3D (b, s, h), scales is 3D (b, s, h/128). If the data type is float8_e8m0 and the data type of x is float8_e4m3fn or float8_e5m2, the quantization mode is MX quantization. If x is 2D, the shape of scales is 3D (bs, h/64, 2). If x is 3D, the shape of scales is 4D (b, s, h/64, 2). h is in the range [1024, 8192] and is 128-aligned. Empty tensors are not supported for this parameter.

- hcom (str): Mandatory parameter. Handle name of the communicator, obtained through the get_hccl_comm_name API.

- world_size (int): Mandatory parameter. Total number of ranks in the communicator. Supports 2, 4, or 8 devices.

- *: Variables before * are position-related and must be entered in order. Variables after * are optional parameters, which are position-independent and must be assigned as key-value pairs. If not assigned, the default value is used.

- reduce_op (str): Optional parameter. Type of the reduce operation. Default value is "sum". Only "sum" is supported in the current version.

- output_dtype (int): Optional parameter. Data type of the output parameter output. Current value: bfloat16, float16, or float32. Default value is bfloat16.

- x_dtype (int): Optional parameter. Data type of the input parameter x. Current value: float8_e4m3fn, float8_e5m2, int8, or hifloat8. Default value is empty.

- scales_dtype (int): Optional parameter. Data type of the quantization coefficients of the data to be communicated. Current value: float32 or float8_e8m0. Default value is empty.

## Return Values

### quant_reduce_scatter

- y (Tensor): Final communication result. Supported data types: float16, bfloat16, and float32. Shape: (b*s/world_size, h). Supported data format: ND.

## Constraints

- This API is supported for training use cases.

- Communication engine constraints:
  - Ascend 950PR&950DT products: Only UB-Memory communication is supported.

- The communicator size supports 2, 4, and 8.

- Communication domain usage constraints:
  - Only the `aclnnQuantAllReduce` and `aclnnQuantReduceScatter` operators can be executed consecutively in the same communicator, and no other communication operators are allowed in the communicator.
  - `HCCL_BUFFSIZE`: Before calling this operator, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communicator, in MB. It defaults to 200 MB when not configured. The requirement is `HCCL_BUFFSIZE` >= 2 * (`xDataSize` + `scalesDataSize + 1`). Here, `xDataSize` is the data size of the input `x`, with the formula: `xDataSize = BS * H * 1 (Byte)`. `scalesDataSize` is the data size of `scales`. When the quantization mode is pertoken-pergroup quantization, the formula is: `scalesDataSize = BS * H / 128 * 4 (Byte)`. When the quantization mode is mx quantization, the formula is: `scalesDataSize = BS * H / 32 * 1 (Byte)`.

- Data type constraints:

    <table style="undefined;table-layout: fixed; width: 1024px"><colgroup>
    <col style="width: 128px">
    <col style="width: 256px">
    <col style="width: 128px">
    <col style="width: 256px">
    </colgroup>
    <thead>
    <tr>
        <th>Use case</th>
        <th>x</th>
        <th>scales</th>
        <th>Y</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>pertoken-pergroup quantization</td>
        <td>hifloat8, int8, float8_e4m3fn, float8_e5m2</td>
        <td>float32</td>
        <td>float16, bfloat16, float32</td>
    </tr>
    <tr>
        <td>MX quantization</td>
        <td>float8_e4m3fn, float8_e5m2</td>
        <td>float8_e8m0</td>
        <td>float16, bfloat16, float32</td>
    </tr>
    </tbody></table>

- Shape constraints:

    <table style="undefined;table-layout: fixed; width: 1024px"><colgroup>
    <col style="width: 128px">
    <col style="width: 256px">
    <col style="width: 256px">
    </colgroup>
    <thead>
    <tr>
        <th>Use Case</th>
        <th>x</th>
        <th>scales</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>pertoken-pergroup quantization</td>
        <td>2D tensor, shape (bs, h).</td>
        <td>2D tensor, shape (bs, h/128).</td>
    </tr>
    <tr>
        <td>pertoken-pergroup quantization</td>
        <td>3D tensor, shape (b, s, h).</td>
        <td>3D tensor, shape (b, s, h/128).</td>
    </tr>
    <tr>
        <td>MX quantization</td>
        <td>2D tensor, shape (bs, h).</td>
        <td>3D tensor, shape (bs, h/64, 2).</td>
    </tr>
    <tr>
        <td>MX quantization</td>
        <td>3D tensor, shape (b, s, h).</td>
        <td>4D tensor, shape (b, s, h/64, 2).</td>
    </tr>
    </tbody></table>

  - The range of H supports only [1024, 8192] and must be 128-aligned.

## Deterministic Computation

- Defaults to a deterministic implementation.

## Example

- Single-operator mode call:

  <!-- npu="950" id7 -->
  - **Ascend 950PR&950DT products**:

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import numpy as np
    from en_dtypes import float8_e8m0
    from cann_ops_transformer.ops import quant_reduce_scatter

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
        output = quant_reduce_scatter(
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

- Graph mode call:

  <!-- npu="950" id8 -->
  - **Ascend 950PR&950DT products**:

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import numpy as np
    from cann_ops_transformer.ops import quant_reduce_scatter

    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    # npu_backend = tng.get_npu_backend(compiler_config=config)
    npu_backend = tng.get_npu_backend(compiler_config=None)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, scales, hcom, world_size, reduce_op="sum", output_dtype=None, x_dtype=None, scales_dtype=None):
            return quant_reduce_scatter(
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
        model = torch.compile(cpu_model, backend=npu_backend, dynamic=False, fullgraph=False)
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
