# QuantReduceScatter

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         |  Supported   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950DT products</term>                             |    √    |
| <term>Atlas A3 products</term>       |    ×    |
| <term>Atlas A2 products</term> |    ×    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                               |    ×    |
| <term>Atlas training products</term>                              |    ×    |

**NOTE:** When using this interface, ensure that both the driver firmware package and the CANN package are of the matched 9.2.0 version or a later matched version. Otherwise, errors such as BUS ERROR will occur.

## Function Description

- Operator function: The QuantReduceScatter operator fuses the ReduceScatter communication of quantized data with the dequantization computation. It first completes the distribution communication of the quantized data and scale factors among ranks according to the ReduceScatter semantics, then dequantizes the data from different ranks element-wise and reduces (sums) them, and outputs the dequantized result of the shard owned by this rank, achieving overlapping of communication and computation.

- Computation formulas:

  - Inputs:
    - $\mathbf{X}_r \in \mathbb{R}^{\text{BS} \times \text{H}}$: the quantized input data of rank $r$, corresponding to the input `x`. $\text{BS}$ is the number of tokens (when `x` is 3-dimensional, the axes are merged as $\text{BS}=\text{b}\times\text{s}$), and $\text{H}$ is the hidden dimension.
    - $\mathbf{S}_r$: the quantization scale factors of rank $r$, corresponding to the input `scales`. The shape depends on the quantization mode; see the parameter description for details.
  - Outputs:
    - $\mathbf{Y}_r \in \mathbb{R}^{\text{BS}' \times \text{H}}$: the final output of rank $r$, corresponding to the output `output`, where $\text{BS}' = \text{BS} / \text{world\_size}$.
  - Conventions:
    - $R$ denotes the total number of ranks in the communication domain, corresponding to the input `world_size`, and $r \in \{0, 1, \dots, R-1\}$.
    - $⊙$ denotes element-wise multiplication; the scale factors are broadcast along the $\text{H}$ axis per quantization group.
    - $\lfloor z \rfloor$ denotes the floor of $z$.
    - $\mathrm{DQ}(\cdot)$ denotes the element-wise dequantization operation performed according to the quantization mode.
  - Overall computation:

    $$
    \mathbf{Y}_r = \left(\sum_{q=0}^{R-1} \mathrm{DQ}\left(\mathbf{X}_q \odot \mathbf{S}_q\right)\right)\left[\, r \cdot \text{BS}' : (r+1) \cdot \text{BS}',\, :\,\right]
    $$

    That is, the quantized data of each rank is first dequantized, then summed element-wise across ranks, and finally split by rows into the output shard of this rank under the ReduceScatter semantics. `reduce_op` only supports "sum" currently.

- Computation process

    The input types and dequantization group granularity of each quantization scenario are as follows:

    | Quantization mode | x | scales | Group granularity | Output |
    | --- | :---: | :---: | :---: | :---: |
    | pertoken-pergroup (KG) quantization | INT8, HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2 | FLOAT32 | One scale shared by every 128 consecutive elements of each token | BFLOAT16, FLOAT16, FLOAT32 |
    | MX quantization | FLOAT8_E4M3FN, FLOAT8_E5M2 | FLOAT8_E8M0 | One scale shared by every 32 consecutive elements | BFLOAT16, FLOAT16, FLOAT32 |

    <details>
    <summary> pertoken-pergroup (KG) quantization scenario</summary>

    - **Data distribution (ReduceScatter communication)**

        Each rank $r$ writes its complete $\mathbf{X}_r$ and $\mathbf{S}_r$ into the data region of the communication windows (Win regions) of all peer ranks, and writes the synchronization flag into the status region; each rank subsequently consumes only the rows related to its own output shard, so the communication is performed as full-volume writing and on-demand reading.

    - **Dequantization and reduce summation**

        This rank reads, from the Win region of each rank $q$, the data blocks belonging to its own output shard. Type conversion is performed first: INT8 is converted to FLOAT32 via FLOAT16, and HIFLOAT8, FLOAT8_E4M3FN and FLOAT8_E5M2 are converted to FLOAT32 directly; the scale factors are FLOAT32 and are broadcast once per 128 consecutive elements to obtain the element-wise scale view. Then the data are dequantized element-wise and accumulated across ranks:

        $$
        \mathbf{D}_q[i,j] = \mathbf{X}_q[i,j] \cdot \mathbf{S}_q\left[i, \left\lfloor j / 128 \right\rfloor\right]
        $$

        $$
        \mathbf{O}[i,j] = \sum_{q=0}^{R-1} \mathbf{D}_q[i,j]
        $$

        The accumulation is performed in FLOAT32, and finally the result is rounded according to `output_dtype` and the shard of this rank $\mathbf{Y}_r = \mathbf{O}\left[\, r \cdot \text{BS}' : (r+1) \cdot \text{BS}',\, :\,\right]$ is written out.
    </details>

    <details>
    <summary> MX quantization scenario</summary>

    - **Data distribution (ReduceScatter communication)**

        Same as the KG scenario: each rank writes its $\mathbf{X}_r$ (FLOAT8_E4M3FN or FLOAT8_E5M2) and $\mathbf{S}_r$ (FLOAT8_E8M0) into the Win regions of all peer ranks and synchronizes the status flags.

    - **Dequantization and reduce summation**

        This rank reads, from the Win region of each rank $q$, the data blocks belonging to its own output shard. The shape of $\mathbf{S}_r$ is $(\text{BS}, \text{H}/64, 2)$ (or $(\text{b}, \text{s}, \text{H}/64, 2)$ when `x` is 3-dimensional), containing $\text{BS} \times \text{H}/32$ scale factors in total, that is, one FLOAT8_E8M0 scale is shared by every 32 consecutive elements. The scale factors are first converted to FLOAT32 via BFLOAT16, and then broadcast per 32 elements to obtain the element-wise scale view:

        $$
        \mathbf{D}_q[i,j] = \mathbf{X}_q[i,j] \cdot \widetilde{\mathbf{S}}_q\left[i, \left\lfloor j / 32 \right\rfloor\right]
        $$

        where $\widetilde{\mathbf{S}}_q$ denotes the per-group scale factors obtained by expanding the packed layout above row by row. Then, same as the KG scenario, the results are accumulated across ranks in FLOAT32, rounded according to `output_dtype`, and the shard of this rank is written out.
    </details>

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 140px">
 <col style="width: 120px">
 <col style="width: 500px">
 <col style="width: 220px">
 <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th>Parameter Name</th>
   <th>Input/Output/Attribute</th>
   <th>Description</th>
   <th>Data Type</th>
   <th>Data Format</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>context</td>
   <td>Input</td>
   <td>MTE communication context data of this rank, containing the address table of the communication domain and the communication window information. The aclnn interface passes this tensor directly; the PyTorch interface builds it automatically from the <code>hcom</code> communication domain name, and users do not need to prepare it explicitly.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x</td>
   <td>Input</td>
   <td>The quantized data of this rank to be communicated. The shape supports 2 dimensions (bs, h) and 3 dimensions (b, s, h); h is 128-aligned and within the range [1024, 8192]. This parameter does not support empty tensors. For the data types supported in each quantization scenario, see the constraints.</td>
   <td>INT8, HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>scales</td>
   <td>Input</td>
   <td>The quantization scale factors of the communicated data. In the pertoken-pergroup (KG) quantization mode, the data type is FLOAT32 and the number of dimensions is the same as x: when x is 2-dimensional, the shape is (bs, h/128); when x is 3-dimensional, the shape is (b, s, h/128). In the MX quantization mode, the data type is FLOAT8_E8M0: when x is 2-dimensional, the shape is (bs, h/64, 2); when x is 3-dimensional, the shape is (b, s, h/64, 2). This parameter does not support empty tensors.</td>
   <td>FLOAT32, FLOAT8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>world_size</td>
   <td>Attribute</td>
   <td>The total number of ranks in the communication domain. 2, 4 and 8 are supported.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>hccl_buffer_size</td>
   <td>Attribute</td>
   <td>The size of the communication buffer occupied by a single communication domain, in MB. The PyTorch interface obtains and passes it automatically according to the communication domain configuration; when calling the aclnn interface directly, it must satisfy the HCCL_BUFFSIZE requirement in the constraints.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>reduce_op</td>
   <td>Optional attribute</td>
   <td>The type of the reduce operation. Default value: "sum". Only "sum" is supported in the current version.</td>
   <td>STRING</td>
   <td></td>
  </tr>
  <tr>
   <td>output_dtype</td>
   <td>Optional attribute</td>
   <td>Specifies the data type of the output parameter. Current values: bfloat16, float16, float32. Default value: bfloat16.</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>output</td>
   <td>Output</td>
   <td>The final communication result, with shape (b*s/world_size, h), always 2-dimensional. The data type is specified by output_dtype, defaulting to bfloat16.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

**NOTE:** The PyTorch interface also provides the optional parameters `x_dtype` and `scales_dtype`, which are used to annotate the real data types carried by uint8 tensors for custom data types (such as HIFLOAT8 and FLOAT8_E8M0); the `hcom` parameter is the communication domain handle name, obtained through the get_hccl_comm_name interface.

## Constraints

- **Usage scenario constraints**:
  - This interface supports training scenarios, currently only <term>Ascend 950DT products</term> are supported.

- **Communication engine constraints**:
  - <term>Ascend 950DT products</term>: only UB-Memory communication is supported.

- **Communication domain constraints**:
  - The communication domain size supports 2, 4 and 8.
  - Within the same communication domain, only the `aclnnQuantAllReduce` and `aclnnQuantReduceScatter` operators are allowed to be executed consecutively, and no other communication operators are allowed in this communication domain.
  - `HCCL_BUFFSIZE`: Before calling this operator, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communication domain, in MB, defaulting to 200 MB when not configured. It is required that `HCCL_BUFFSIZE` >= 2 * CeilDiv(`xDataSize` + `scalesDataSize` + `reserveSize`, 1024 * 1024), where CeilDiv is the ceiling division and the right-hand side is expressed in MB. All terms are calculated in bytes: `xDataSize` is the data size of the input `x`, `xDataSize = BS * H * 1 (Byte)`; `scalesDataSize` is the data size of `scales`, `BS * H / 128 * 4 (Byte)` for pertoken-pergroup quantization and `BS * H / 32 * 1 (Byte)` for MX quantization; `reserveSize = 1 * 1024 * 1024 (Byte)` is reserved for the status region.

- **Parameter constraints**:
  - The BS axis of `x` (the merged B*S axes for 3-dimensional input) must be divisible by `world_size`.
  - The h axis of `x` only supports the range [1024, 8192] and must be 128-aligned.
  - Neither `x` nor `scales` supports empty tensors.
  - `output` is always 2-dimensional, with shape (BS/world_size, h).
  - The data format of `x`, `scales` and `output` only supports ND.

- **Data type constraints**:

    <table style="undefined;table-layout: fixed; width: 1024px"><colgroup>
    <col style="width: 256px">
    <col style="width: 384px">
    <col style="width: 128px">
    <col style="width: 256px">
    </colgroup>
    <thead>
    <tr>
        <th>Scenario</th>
        <th>x</th>
        <th>scales</th>
        <th>output</th>
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

- **Shape constraints**:

    <table style="undefined;table-layout: fixed; width: 1024px"><colgroup>
    <col style="width: 256px">
    <col style="width: 256px">
    <col style="width: 256px">
    </colgroup>
    <thead>
    <tr>
        <th>Scenario</th>
        <th>x</th>
        <th>scales</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>pertoken-pergroup quantization</td>
        <td>2-dimensional tensor, shape (bs, h)</td>
        <td>2-dimensional tensor, shape (bs, h/128)</td>
    </tr>
    <tr>
        <td>pertoken-pergroup quantization</td>
        <td>3-dimensional tensor, shape (b, s, h)</td>
        <td>3-dimensional tensor, shape (b, s, h/128)</td>
    </tr>
    <tr>
        <td>MX quantization</td>
        <td>2-dimensional tensor, shape (bs, h)</td>
        <td>3-dimensional tensor, shape (bs, h/64, 2)</td>
    </tr>
    <tr>
        <td>MX quantization</td>
        <td>3-dimensional tensor, shape (b, s, h)</td>
        <td>4-dimensional tensor, shape (b, s, h/64, 2)</td>
    </tr>
    </tbody></table>

- **Deterministic computation**:
  - Deterministic computation is supported by default.

- **Reserved and non-public parameters**:
  - Some parameters and data types in the parameter table are not yet publicly available and are reserved or used for internal implementation. The introduction to the interface parameters and their constraints are described in detail in the interface document [quant_reduce_scatter Operator Interface Document](docs/torchapi_quant_reduce_scatter_en.md).

## Invocation Description

| Invocation Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| PyTorch interface invocation | - | Invoke the quant_reduce_scatter operator through the [quant_reduce_scatter](docs/torchapi_quant_reduce_scatter_en.md) PyTorch interface. Both single-operator mode and torchair graph mode are supported. |
