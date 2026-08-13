# low_latency_dispatch

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
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

  需与[low_latency_combine](low_latency_combine.md)配套使用，完成MoE的并行部署下的token的dispatch和combine。
  - 支持非量化、静态量化、pertoken动态量化、pergroup动态量化、mx动态量化和mx clip动态量化场景，对token数据先进行量化（可选），进行EP（Expert Parallelism）域的alltoallv通信；
  - 支持特殊专家场景。

- **计算公式**：

  - 量化场景：

      若`x_smooth_scale`不为None，先按专家维或量化系数进行平滑/缩放；若为None，则直接使用`x`：

      $$
      x\_fp32 =
      \begin{cases}
      CastToFp32(x) \times x\_smooth\_scale, & \quad \text{if } x\_smooth\_scale \ne None \\
      CastToFp32(x), & \quad \text{if } x\_smooth\_scale = None
      \end{cases}
      $$

      如果quant_mode=0（非量化）

      $$
      quant\_out = x
      $$

      $$
      alltoall\_x\_out = alltoallv(quant\_out)
      $$

      $$
      expand\_x = alltoall\_x\_out
      $$

      如果quant_mode=1（静态量化）

      $$
      quant\_out = Cast(CastToFp32(x) \times scales, dstType)
      $$

      $$
      alltoall\_x\_out = alltoallv(quant\_out)
      $$

      $$
      expand\_x = alltoall\_x\_out
      $$

      如果quant_mode=2（动态量化）

      $$
      x\_fp32 = CastToFp32(x) \times scales
      $$

      $$
      dynamic\_scales\_value = dst\_type\_max / Max(Abs(x\_fp32))
      $$

      $$
      quant\_out = Cast(x\_fp32 \times dynamic\_scales\_value, dstType)
      $$

      $$
      alltoall\_x\_out = alltoallv(quant\_out)
      $$

      $$
      alltoall\_dynamic\_scales\_out = alltoallv(1.0 / dynamic\_scales\_value)
      $$

      $$
      expand\_x = alltoall\_x\_out
      $$

      $$
      dynamic\_scales = alltoall\_dynamic\_scales\_out
      $$

      如果quant_mode=3（pergroup量化）

      $$
      x\_fp32 = CastToFp32(x) \times scales
      $$

      $$
      dynamic\_scales\_value = dst\_type\_max / Max(Abs(x\_fp32))
      $$

      $$
      quant\_out = Cast(x\_fp32 \times dynamic\_scales\_value, dstType)
      $$

      $$
      alltoall\_x\_out = alltoallv(quant\_out)
      $$

      $$
      alltoall\_dynamic\_scales\_out = alltoallv(1.0 / dynamic\_scales\_value)
      $$

      $$
      expand\_x = alltoall\_x\_out
      $$

      $$
      dynamic\_scales = alltoall\_dynamic\_scales\_out
      $$

      如果quant_mode=4（mxfp8量化）

      $$
      shared\_exp = floor(log_2(max(x))) - emax
      $$

      $$
      dynamic\_scales\_value = 2^{shared\_exp}
      $$

      $$
      quant\_out = Cast(x / dynamic\_scales\_value, dstType)
      $$

      $$
      alltoall\_x\_out = alltoallv(quant\_out)
      $$

      $$
      alltoall\_dynamic\_scales\_out = alltoallv(1.0 / dynamic\_scales\_value)
      $$

      $$
      expand\_x = alltoall\_x\_out
      $$

      $$
      dynamic\_scales = alltoall\_dynamic\_scales\_out
      $$

      如果quant_mode=5（mxfp8量化chip方法）

      $$
      max\_abs = max(abs(x))
      $$

      $$
      max\_abs\_clamp = max(max\_abs, 10^{-4})
      $$

      $$
      shared\_exp = ceil(log_2(max\_abs\_clamp / fp8\_max))
      $$

      $$
      dynamic\_scales\_value = 2^{shared\_exp}
      $$

      $$
      quant\_out = CastToFp8(x / dynamic\_scales\_value)
      $$

      $$
      alltoall\_x\_out = alltoallv(quant\_out)
      $$

      $$
      alltoall\_dynamic\_scales\_out = alltoallv(1.0 / dynamic\_scales\_value)
      $$

      $$
      expand\_x = alltoall\_x\_out
      $$

      $$
      dynamic\_scales = alltoall\_dynamic\_scales\_out
      $$

  - 特殊专家场景：

      零专家场景，即`zero_Expert_Num`不为0：

      $$
      Moe(ori\_x) = 0
      $$

      拷贝专家场景，即`copy_Expert_Num`不为0：

      $$
      Moe(ori\_x) = ori\_x
      $$

      常量专家场景，即`const_expert_num`不为0：

      $$
      Moe(ori\_x) = const\_expert\_alpha\_1 * ori\_x + const\_expert\_alpha\_2 * const\_expert\_v
      $$

      参数ori\_x、const\_expert\_alpha\_1、const\_expert\_alpha\_2、const\_expert\_v见[low_latency_combine](low_latency_combine.md)文档。

## 函数原型

```python
MoeDistributeBuffer.low_latency_dispatch(x, topk_idx, num_experts, *, quant_mode=0, comm_alg="", x_smooth_scale=None, x_active_mask=None, topk_weights=None, zero_expert_num=0, copy_expert_num=0, const_expert_num=0, elastic_info=None, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, expert_token_nums_type=1, num_max_dispatch_tokens_per_rank=0, y_dtype=None, x_dtype=None, x_smooth_scales_dtype=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>表示计算使用的token数据，需根据`topk_idx`来发送给其他卡。要求为2维张量，表示有BS个token，数据格式为ND，支持非连续的Tensor。</td>
        <td>bfloat16、float16、uint8、float8_e5m2、float8_e4m3fn</td>
        <td>(BS, H)</td>
    </tr>
    <tr>
        <td>topk_idx</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>表示每个token的topK个专家索引，决定每个token要发给哪些专家。要求为2维张量，数据格式为ND，支持非连续的Tensor。对应[low_latency_combine](low_latency_combine.md)的`topk_idx`输入，张量里value取值范围为[0, num_experts)，且同一行中的K个value不能重复。</td>
        <td>int32</td>
        <td>(BS, K)</td>
    </tr>
    <tr>
        <td>num_experts</td>
        <td>int</td>
        <td>必选</td>
        <td>MoE专家数量，取值范围[1, 1024]，并且满足以下条件：num_experts%(ep_world_size - shared_expert_rank_num)=0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quant_mode</td>
        <td>int</td>
        <td>可选</td>
        <td>表示量化模式，0表示非量化（默认），1表示静态量化，2表示pertoken动态量化，3表示pergroup动态量化，4表示mx动态量化，5表示mx clip动态量化。`quant_mode`为2、3、4、5时，`dynamic_scales`不为None；`quant_mode`为0且输入`x`为hifloat8、float8_e5m2、float8_e4m3fn、float4_e2m1、float4_e1m2等低精度类型时，`dynamic_scales`也不为None；其他场景下`dynamic_scales`为None。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>comm_alg</td>
        <td>str</td>
        <td>可选</td>
        <td>表示通信亲和内存布局算法。当前版本支持""，"fullmesh_v1"，"fullmesh_v2"三种输入方式。""为默认值，开启fullmesh_v1模板；"fullmesh_v1"开启fullmesh_v1模板；"fullmesh_v2"开启fullmesh_v2模板。</td>
        <td>str</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x_smooth_scale</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>
            <ul>
                <li>表示量化平滑参数或量化系数。</li>
                <li>`quant_mode`为0时，若`x`为bfloat16或float16则不传，若`x`为hifloat8、float8_e5m2、float8_e4m3fn、float4_e2m1、float4_e1m2等低精度类型则必须传有效Tensor。</li>
                <li>`quant_mode`为1时必须传。</li>
                <li>`quant_mode`为2或3时可传可不传。</li>
                <li>`quant_mode`为4或5时不传。</li>
                <li>若用于专家维平滑，要求为2维张量，如果有共享专家，shape为(shared_expert_num+num_experts, H)，如果没有共享专家，shape为(num_experts, H)，数据格式为ND，不支持非连续的Tensor。</li>
                <li>其他量化系数场景的shape约束见“quant_mode相关约束”。</li>
            </ul>
        </td>
        <td>float</td>
        <td>
            <ul>
                <li>(shared_expert_num+num_experts, H)</li>
                <li>(num_experts, H)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>x_active_mask</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>表示token是否参与通信，要求是一个1维或者2维张量。当输入为1维时，shape为(BS, )；当输入为2维时，shape为(BS, K)。数据格式要求为ND，支持非连续的Tensor。当输入为1维时，参数为true表示对应的token参与通信，true必须排到false之前，例：{true, false, true}为非法输入；当输入为2D时，参数为true表示当前token对应的`topk_idx`参与通信，若当前token对应的K个`bool`值全为false，表示当前token不会参与通信。默认所有token都会参与通信。当每张卡的BS数量不一致时，所有token必须全部有效。</td>
        <td>bool</td>
        <td>
            <ul>
                <li>(BS, )</li>
                <li>(BS, K)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>topk_weights</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>表示每个token的topK个专家权重。</td>
        <td>float32</td>
        <td>(BS, K)</td>
    </tr>
    <tr>
        <td>zero_expert_num</td>
        <td>int</td>
        <td>可选</td>
        <td>表示零专家的数量，取值范围[0, MAX_int32)，MAX_int32 = 2^31 - 1，合法的零专家的ID值是[num_experts, num_experts+zero_expert_num)。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>copy_expert_num</td>
        <td>int</td>
        <td>可选</td>
        <td>表示拷贝专家的数量，取值范围[0, MAX_int32)，MAX_int32 = 2^31 - 1，合法的拷贝专家的ID值是[num_experts+zero_expert_num, num_experts+zero_expert_num+copy_expert_num)。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>const_expert_num</td>
        <td>int</td>
        <td>可选</td>
        <td>表示常量专家的数量，取值范围[0, MAX_int32)，MAX_int32 = 2^31 - 1，合法的常量专家的ID值是[moe_expert_num+zero_expert_num+copy_expert_num, moe_expert_num+zero_expert_num+copy_expert_num+const_expert_num)。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>elastic_info</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>预留参数，当前版本不支持，传默认值None即可。</td>
        <td>暂不支持</td>
        <td>-</td>
    </tr>
    <tr>
        <td>expert_shard_type</td>
        <td>int</td>
        <td>可选</td>
        <td>表示共享专家卡排布类型。当前仅支持0，表示共享专家卡排在MoE专家卡前面。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>shared_expert_num</td>
        <td>int</td>
        <td>可选</td>
        <td>表示共享专家数量，一个共享专家可以复制部署到多个卡上。取值范围[0, 4]，0表示无共享专家，默认值为1。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>shared_expert_rank_num</td>
        <td>int</td>
        <td>可选</td>
        <td>表示共享专家卡数量。取值范围[0, ep_world_size)。取0表示无共享专家，不取0需满足shared_expert_rank_num%shared_expert_num=0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>expert_token_nums_type</td>
        <td>int</td>
        <td>可选</td>
        <td>表示输出`expert_token_nums`的值类型，取值范围[0, 1]，0表示每个专家收到token数量的前缀和，1表示每个专家收到的token数量（默认）。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_max_dispatch_tokens_per_rank</td>
        <td>int</td>
        <td>可选</td>
        <td>表示每张卡上的token数量。当每个rank的BS不同时，最大的BS大小，当每个rank上BS相同时，默认为0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>y_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>
            <ul>
                <li>表示`expand_x`输出的逻辑数据类型，默认值为None，由`quant_mode`和输入类型推导。</li>
                <li>Ascend 950场景可通过该参数指定低精度输出类型。</li>
                <li>`quant_mode=1`支持int8、hifloat8。</li>
                <li>`quant_mode=2`支持int8、float8_e4m3fn、float8_e5m2。</li>
                <li>`quant_mode=3`支持float8_e4m3fn、float8_e5m2。</li>
                <li>`quant_mode=4`或`quant_mode=5`支持float8_e4m3fn、float8_e5m2、float4_e2m1、float4_e1m2。</li>
                <li>传值需使用`cann_ops_transformer.common`中与ACL数据类型对应的整数枚举。</li>
            </ul>
        </td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>表示输入`x`的逻辑数据类型，默认值为None，由`x`的PyTorch存储类型推导。低精度场景下可指定为hifloat8、float8_e5m2、float8_e4m3fn、float4_e2m1、float4_e1m2等ACL数据类型；其中hifloat8、float4_e2m1、float4_e1m2在PyTorch侧以uint8存储。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x_smooth_scales_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>表示`x_smooth_scale`的逻辑数据类型，默认值为None，由`x_smooth_scale`的PyTorch存储类型推导。低精度场景下可指定float8_e8m0等ACL数据类型；例如`x`为float8_e5m2或float8_e4m3fn时，`x_smooth_scale`支持float或float8_e8m0，`x`为float4_e2m1或float4_e1m2时，`x_smooth_scale`要求float8_e8m0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## 返回值说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>expand_x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>表示本卡收到的token数据，要求为2维张量，A表示在EP通信域可能收到的最大token数。非量化且未指定`y_dtype`时与`x`数据类型保持一致；量化或指定`y_dtype`时类型由`quant_mode`和`y_dtype`决定，支持int8、hifloat8、float8_e4m3fn、float8_e5m2、float4_e2m1、float4_e1m2等低精度输出。数据格式为ND，支持非连续的Tensor。</td>
        <td>bfloat16、float16、int8、uint8、float8_e5m2、float8_e4m3fn</td>
        <td>(A, H)；float4_e2m1/float4_e1m2时为(A, H/2)</td>
    </tr>
    <tr>
        <td>dynamic_scales</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>
            <ul>
                <li>表示计算得到的动态量化参数。</li>
                <li>`quant_mode`为2、3、4、5时有输出。</li>
                <li>`quant_mode`为0且输入`x`为hifloat8、float8_e5m2、float8_e4m3fn、float4_e2m1、float4_e1m2等低精度类型时也有输出。</li>
                <li>float8_e8m0在PyTorch侧用uint8承载。</li>
                <li>数据格式支持ND，支持非连续的Tensor。</li>
            </ul>
        </td>
        <td>float、uint8</td>
        <td>
            <ul>
                <li>pertoken动态量化：(A,)</li>
                <li>pergroup动态量化：(A, Ceil(H, 128))</li>
                <li>mx动态量化/mx clip动态量化：(A, Align2(Ceil(H, 32)))</li>
                <li>低精度非量化输入：(A, dim1)，dim1由`x_smooth_scale`第二维决定</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>assist_info_for_combine</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>表示给同一专家发送的token个数，要求是一个1维张量，数据格式为ND，支持非连续的Tensor。对应[low_latency_combine](low_latency_combine.md)的`assist_info_for_combine`输入。</td>
        <td>int32</td>
        <td>(A * 128, )</td>
    </tr>
    <tr>
        <td>expert_token_nums</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>本卡每个专家实际收到的token数量，要求为1维张量，数据格式支持ND，支持非连续的Tensor。</td>
        <td>int64</td>
        <td>(local_expert_num,)</td>
    </tr>
    <tr>
        <td>ep_recv_counts</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>表示EP通信域各卡收到的token数（token数以前缀和的形式表示），要求为1维张量，数据格式支持ND，支持非连续的Tensor。对应[low_latency_combine](low_latency_combine.md)的`ep_send_counts`输入。</td>
        <td>int32</td>
        <td>(ep_world_size*local_expert_num, )</td>
    </tr>
    <tr>
        <td>expand_scales</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>表示`topk_weights`与`x`一起进行alltoallv之后的输出。</td>
        <td>float32</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和torchair图模式调用，`low_latency_dispatch`和`low_latency_combine`必须配套使用。
- 各输入Tensor的list[Tensor]长度、是否转置、是否支持非连续Tensor约束如下：

    <table style="undefined;table-layout: fixed; width:800px"><colgroup>
    <col style="width: 200px">
    <col style="width: 200px">
    <col style="width: 150px">
    <col style="width: 250px">
    </colgroup>
    <thead>
    <tr>
        <th>参数名</th>
        <th>list[Tensor]长度</th>
        <th>是否转置</th>
        <th>是否支持非连续Tensor</th>
    </tr>
    </thead>
    <tbody>
        <tr>
            <td>x</td>
            <td>-</td>
            <td>否</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>topk_idx</td>
            <td>-</td>
            <td>否</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>x_smooth_scale</td>
            <td>-</td>
            <td>否</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>x_active_mask</td>
            <td>-</td>
            <td>否</td>
            <td>支持</td>
        </tr>
    </tbody>
    </table>

- 参数里Shape使用的变量如下：
  - A：表示本卡接收的最大token数量，取值范围如下
    - 对于共享专家，要满足A=BS\*shared\_expert\_num/shared\_expert\_rank\_num。
    - 对于MoE专家，当`num_max_dispatch_tokens_per_rank`为0时，要满足A \>= BS \* ep\_world\_size \* min\(local\_expert\_num, K\)；当`num_max_dispatch_tokens_per_rank`不为0时，要满足A \>= num_max\_dispatch\_tokens\_per\_rank \* ep\_world\_size \* min\(local\_expert\_num, K\)。

  - H：表示hidden size隐藏层大小。取值为\[1024, 8192\]。

  - BS：表示batch sequence size，即本卡最终输出的token数量。取值范围为0<BS≤512。

  - K：表示选取topK个专家，取值范围为0<K≤16，同时满足0 < K ≤ num\_experts + zero_expert_num + copy_expert_num + const_expert_num。

  - local\_expert\_num：表示本卡专家数量。
    - 对于共享专家卡，local\_expert\_num为1。
    - 对于MoE专家卡，local\_expert\_num=num\_experts/\(ep\_world\_size-shared\_expert\_rank\_num)，应满足0 < local_expert_num * ep_world_size ≤ 2048。

- 在不同产品型号、不同通信算法或不同版本中，`low_latency_dispatch`的Tensor输出`assist_info_for_combine`、`ep_recv_counts`、`expand_scales`中的元素值可能不同，使用时直接将上述Tensor传给`low_latency_combine`对应参数即可，模型其他业务逻辑不应对其存在依赖。
- 调用接口过程中使用的`num_experts`、`expert_shard_type`、`shared_expert_num`、`shared_expert_rank_num`、`num_max_dispatch_tokens_per_rank`参数取值所有卡需保持一致，`expert_shard_type`、`num_max_dispatch_tokens_per_rank`网络中不同层中也需保持一致，且和[low_latency_combine](low_latency_combine.md)对应参数也保持一致。
- 该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
- num_experts + zero_expert_num + copy_expert_num + const_expert_num < MAX_int32。
- 相关约束：
  <!-- npu="910b" id7 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - `quant_mode=0`表示非量化场景，`x_smooth_scale`不传，`expand_x`的数据类型支持float16、bfloat16。
    - `quant_mode=2`表示pertoken动态量化场景，`expand_x`的数据类型支持int8，`x_smooth_scale`可传可不传；若传入有效Tensor，shape为(num_experts, H)，输出`dynamic_scales` shape为(A,)。
  <!-- end id7 -->
  <!-- npu="A3" id8 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 不支持topK专家权重功能，`topk_weights`必须传None，`expand_scales`输出无效。
    - `quant_mode=0`表示非量化场景，`x_smooth_scale`不传，`expand_x`的数据类型支持float16、bfloat16。
    - `quant_mode=2`表示pertoken动态量化场景，`expand_x`的数据类型支持int8，`x_smooth_scale`可传可不传；若存在共享专家且传入有效Tensor，shape为(shared_expert_num + num_experts, H)，若不存在共享专家，shape为(num_experts, H)，输出`dynamic_scales` shape为(A,)。
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - `topk_weights`可传有效Tensor或None。传有效Tensor时使能topK专家权重功能，有效Tensor为2D Tensor，shape为(BS, K)，且`expand_scales`输出有效；传None时不使能topK专家权重功能，`expand_scales`输出无效。不支持空Tensor。
    - `quant_mode=0`表示非量化场景。当`x`为float16或bfloat16时，`expand_x`可与`x`一致，也可通过`y_dtype`指定为hifloat8，`x_smooth_scale`必须为None；当`x`为hifloat8、float8_e5m2、float8_e4m3fn、float4_e2m1、float4_e1m2时，`x_smooth_scale`必须传有效Tensor，`expand_x`与`x`逻辑类型一致。`x`为hifloat8时，`x_smooth_scale`逻辑类型为float；`x`为float8_e5m2或float8_e4m3fn时，`x_smooth_scale`逻辑类型为float或float8_e8m0；`x`为float4_e2m1或float4_e1m2时，`x_smooth_scale`逻辑类型为float8_e8m0，且H必须为偶数。此时`x_smooth_scale` shape为(BS, dim1)，其中dim1 <= H。
    - `quant_mode=1`表示静态量化场景，`expand_x`支持int8、hifloat8，`x_smooth_scale`必须传有效Tensor。`expand_x`为int8时，`x_smooth_scale`可表示量化系数，shape为(1,)；也可表示每个专家共享的平滑权重，shape为(H,)；也可表示融合了每个专家平滑权重的量化系数，有共享专家时shape为(shared_expert_num + num_experts, H)，无共享专家时shape为(num_experts, H)。`expand_x`为hifloat8时，`x_smooth_scale` shape必须为(1,)。
    - `quant_mode=2`表示pertoken动态量化场景，`expand_x`支持int8、float8_e4m3fn、float8_e5m2，`x_smooth_scale`可传可不传；若传入有效Tensor，其专家维shape规则同`quant_mode=1`，输出`dynamic_scales` shape为(A,)。
    - `quant_mode=3`表示pergroup动态量化场景，`expand_x`支持float8_e4m3fn、float8_e5m2，`x_smooth_scale`可传可不传；若传入有效Tensor，其专家维shape规则同`quant_mode=1`，输出`dynamic_scales` shape为(A, Ceil(H, 128))。
    - `quant_mode=4`表示mx动态量化场景，`quant_mode=5`表示mx clip动态量化场景，`expand_x`支持float8_e4m3fn、float8_e5m2、float4_e2m1、float4_e1m2，`x_smooth_scale`必须为None，输出`dynamic_scales` shape为(A, Align2(Ceil(H, 32)))；当`expand_x`为float4_e2m1或float4_e1m2时，H必须为偶数。
  <!-- end id9 -->
  - 本文中的`Ceil(H, N)`表示`(H + N - 1) // N`，`Align2(x)`表示向上对齐到2的整数倍。
- HCCL通信域缓存区大小:
    调用本接口前需检查HCCL\_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。
  - 该场景仅支持通过环境变量HCCL\_BUFFSIZE配置，该环境变量按通信域粒度管理，每个通信域独占一组“2*HCCL\_BUFFSIZE”大小的内存。
  - ep通信域内，comm\_alg配置为"fullmesh_v1"或"": 设置大小要求 \>= 2 \* \(local\_expert\_num \* max\_bs \* ep\_world\_size \* Align512\(Align32\(2 \* H\) + 64\) + \(K + shared\_expert\_num\) \* max\_bs \* Align512\(2 \* H\)\)。
  - ep通信域内，comm\_alg配置为"fullmesh_v2": 设置大小要求 \>= 2 \* \(local\_expert\_num \* max\_bs \* ep\_world\_size \* 480Align512\(Align32\(2 \* H\) + 64\) + \(K + shared\_expert\_num\) \* max\_bs \* Align512\(2 \* H\)\)。
  - 其中480Align512(x) = ((x+480-1)/480)\*512,Align512(x) = ((x+512-1)/512)\*512,Align32(x) = ((x+32-1)/32)\*32。
  - 通信域开设大小可通过调用MoeDistributeBuffer.get_low_latency_ccl_buffer_size接口计算。

- 本文公式中的“/”表示整除。

- 通信域使用约束：

  - 一个模型中的`low_latency_dispatch`和`low_latency_combine`算子仅支持相同EP通信域，且该通信域中不允许有其他算子。

  - 一个通信域内的节点需在一个超节点内，不支持跨超节点。

- 版本配套约束：

     静态图模式下，从TorchNPU 8.0.0版本开始，TorchNPU框架会对静态图中最后一个节点输出结果做Meta推导与inferShape推导的结果强校验。当图中只有一个Dispatch算子，若CANN版本落后于TorchNPU版本，会出现Shape不匹配报错，建议用户升级CANN版本，详细的版本配套关系参见《[TorchNPU版本说明](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-7.3.0/docs/zh/release_notes/release_notes.md)》中“相关产品版本配套说明”。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  ```python
  import os
  import torch
  import random
  import torch_npu
  import numpy as np
  from torch.multiprocessing import Process
  import torch.distributed as dist
  from torch.distributed import ReduceOp
  from cann_ops_transformer.ops import MoeDistributeBuffer

  # 控制模式
  quant_mode = 2  # 2为动态量化
  is_dispatch_scales = True  # 动态量化可选择是否传scales
  input_dtype = torch.bfloat16  # 输出dtype
  server_num = 1
  server_index = 0
  port = 50001
  master_ip = '127.0.0.1'
  dev_num = 16
  world_size = server_num * dev_num
  rank_per_dev = int(world_size / server_num)  # 每个host有几个die
  shared_expert_rank_num = 0  # 共享专家数
  num_experts = 32  # moe专家数
  bs = 8  # token数量
  h = 7168  # 每个token的长度
  k = 8
  random_seed = 0
  ep_world_size = world_size
  moe_rank_num = ep_world_size - shared_expert_rank_num
  local_moe_expert_num = num_experts // moe_rank_num
  globalBS = bs * ep_world_size
  is_shared = (shared_expert_rank_num > 0)
  is_quant = (quant_mode > 0)
  zero_expert_num = 1
  copy_expert_num = 1
  const_expert_num = 0


  def gen_const_expert_alpha_1():
      const_expert_alpha_1 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
      return const_expert_alpha_1


  def gen_const_expert_alpha_2():
      const_expert_alpha_2 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
      return const_expert_alpha_2


  def gen_const_expert_v():
      const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
      return const_expert_v


  def get_new_group(rank):
      ep_ranks = list(range(ep_world_size))
      ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
      if rank in ep_ranks:
          print(f"rank:{rank} ep_ranks:{ep_ranks}")
      return ep_group


  def get_hcomm_info(rank, comm_group):
      if torch.__version__ > '2.0.1':
          hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
      else:
          hcomm_info = comm_group.get_hccl_comm_name(rank)
      return hcomm_info


  def get_dispatch_kwargs_warmup(
      x_warm_up, topk_idx_warm_up,
  ):
      x_warm_up = x_warm_up.to(input_dtype).npu()
      topk_idx_warm_up = topk_idx_warm_up.to(torch.int32).npu()
      return {
          'x': x_warm_up,
          'topk_idx': topk_idx_warm_up,
          'x_active_mask': None,
          'shared_expert_num': 0,
          'shared_expert_rank_num': shared_expert_rank_num,
          'num_experts': num_experts,
          'quant_mode': 2,
          'num_max_dispatch_tokens_per_rank': 16,
      }


  def run_npu_process(rank):
      torch_npu.npu.set_device(rank)
      rank = rank + 16 * server_index
      # 初始化分布式进程组
      dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
      ep_group = get_new_group(rank)
      ep_hcomm_info = get_hcomm_info(rank, ep_group)

      # 创建输入tensor
      x = torch.randn(bs, h, dtype=input_dtype).npu()
      topk_idx = torch.tensor([[5, 7, 17, 4, 2, 6, 11, 16],
                              [10, 12, 13, 15, 19, 4, 18, 1],
                              [19, 33, 1, 17, 9, 5, 0, 32],
                              [19, 11, 17, 0, 10, 5, 7, 9],
                              [10, 16, 11, 17, 33, 8, 9, 3],
                              [12, 19, 5, 7, 1, 3, 18, 16],
                              [11, 9, 13, 16, 12, 33, 17, 14],
                              [16, 4, 9, 5, 0, 10, 11, 17]], dtype=torch.int32).npu()
      topk_weights = torch.randn(bs, k, dtype=torch.float32).npu()

      scales_shape = (1 + num_experts, h) if shared_expert_rank_num else (num_experts, h)
      if is_dispatch_scales:
          scales = torch.randn(scales_shape, dtype=torch.float32).npu()
      else:
          scales = None

      const_expert_alpha_1 = None
      const_expert_alpha_2 = None
      const_expert_v = None

      # 构造通信域buffer
      distribute_buffer = MoeDistributeBuffer(ep_group)

      # 调用dispatch，完成token的量化与EP域alltoallv分发
      expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_recv_counts, _ = distribute_buffer.low_latency_dispatch(
          x=x,
          topk_idx=topk_idx,
          shared_expert_num=0,
          shared_expert_rank_num=shared_expert_rank_num,
          num_experts=num_experts,
          quant_mode=quant_mode,
          num_max_dispatch_tokens_per_rank=globalBS,
          zero_expert_num=zero_expert_num,
          copy_expert_num=copy_expert_num,
          const_expert_num=const_expert_num)

      if is_quant:
          expand_x = expand_x.to(input_dtype)

      # 将dispatch输出配套传入combine，完成token的回收
      x = distribute_buffer.low_latency_combine(x=expand_x,
                                              topk_idx=topk_idx,
                                              topk_weights=topk_weights,
                                              assist_info_for_combine=assist_info_for_combine,
                                              ep_send_counts=ep_recv_counts,
                                              shared_expert_num=0,
                                              shared_expert_rank_num=shared_expert_rank_num,
                                              num_experts=num_experts,
                                              num_max_dispatch_tokens_per_rank=globalBS,
                                              ori_x=x,
                                              const_expert_alpha_1=const_expert_alpha_1,
                                              const_expert_alpha_2=const_expert_alpha_2,
                                              const_expert_v=const_expert_v,
                                              zero_expert_num=zero_expert_num,
                                              copy_expert_num=copy_expert_num,
                                              const_expert_num=const_expert_num)
      print(f'rank {rank} npu finished! \n')


  if __name__ == "__main__":
      print(f"bs={bs}")
      print(f"num_max_dispatch_tokens_per_rank={globalBS}")
      print(f"shared_expert_rank_num={shared_expert_rank_num}")
      print(f"num_experts={num_experts}")
      print(f"k={k}")
      print(f"quant_mode={quant_mode}", flush=True)
      print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
      print(f"ep_world_size={ep_world_size}", flush=True)

      if shared_expert_rank_num > ep_world_size:
          print("shared_expert_rank_num不能大于ep_world_size")
          exit(0)
      if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
          print("ep_world_size必须是shared_expert_rank_num的整数倍")
          exit(0)
      if num_experts % moe_rank_num != 0:
          print("num_experts必须是moe_rank_num的整数倍")
          exit(0)
      p_list = []
      for rank in range(rank_per_dev):
          p = Process(target=run_npu_process, args=(rank,))
          p_list.append(p)

      for p in p_list:
          p.start()
      for p in p_list:
          p.join()

      print("run npu success.")
  ```

- 图模式调用：

    ```python
    import os
    import torch
    import random
    import torch_npu
    import torchair
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import time
    from cann_ops_transformer.ops import MoeDistributeBuffer

    # 控制模式
    quant_mode = 2  # 2为动态量化
    is_dispatch_scales = True  # 动态量化可选择是否传scales
    input_dtype = torch.bfloat16  # 输出dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # 每个host有几个die
    shared_expert_rank_num = 0  # 共享专家数
    num_experts = 32  # moe专家数
    bs = 8  # token数量
    h = 7168  # 每个token的长度
    k = 8
    random_seed = 0
    ep_world_size = world_size
    moe_rank_num = ep_world_size - shared_expert_rank_num
    local_moe_expert_num = num_experts // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)

    zero_expert_num = 1
    copy_expert_num = 1
    const_expert_num = 0

    class MOE_DISTRIBUTE_GRAPH_Model(torch.nn.Module):
        def __init__(self, group_ep):
            super().__init__()
            self.group = group_ep
            self.distribute_buffer = MoeDistributeBuffer(group_ep)

        def forward(self, x, topk_idx, group_ep,
                    ep_world_size, ep_rank_id, expert_shard_type, shared_expert_rank_num, num_experts,
                    scales, quant_mode, num_max_dispatch_tokens_per_rank, topk_weights, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v, zero_expert_num, copy_expert_num, const_expert_num):
            output_dispatch_npu = self.distribute_buffer.low_latency_dispatch(
                x=x,
                topk_idx=topk_idx,
                shared_expert_num=0,
                shared_expert_rank_num=shared_expert_rank_num,
                num_experts=num_experts,
                num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                elastic_info=elastic_info,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num
            )
            expand_x_npu, _, assist_info_for_combine_npu, _, ep_recv_counts_npu, expand_scales = output_dispatch_npu
            if expand_x_npu.dtype == torch.int8:
                expand_x_npu = expand_x_npu.to(input_dtype)
            output_combine_npu = self.distribute_buffer.low_latency_combine(
                x=expand_x_npu,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                assist_info_for_combine=assist_info_for_combine_npu,
                ep_send_counts=ep_recv_counts_npu,
                shared_expert_num=0,
                shared_expert_rank_num=shared_expert_rank_num,
                num_experts=num_experts,
                num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                elastic_info=elastic_info,
                ori_x=x,
                const_expert_alpha_1=const_expert_alpha_1,
                const_expert_alpha_2=const_expert_alpha_2,
                const_expert_v=const_expert_v,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num
            )
            x = output_combine_npu
            x_combine_res = output_combine_npu
            return [x_combine_res, output_combine_npu]

    def gen_const_expert_alpha_1():
        const_expert_alpha_1 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1

    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_2

    def gen_const_expert_v():
        const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_v

    def get_new_group(rank):
        ep_ranks = list(range(ep_world_size))
        ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
        if rank in ep_ranks:
            print(f"rank:{rank} ep_ranks:{ep_ranks}")
        return ep_group

    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info

    def get_dispatch_kwargs_warmup(
        x_warm_up, topk_idx_warm_up,
    ):
        x_warm_up = x_warm_up.to(input_dtype).npu()
        topk_idx_warm_up = topk_idx_warm_up.to(torch.int32).npu()
        return {
            'x': x_warm_up,
            'topk_idx': topk_idx_warm_up,
            'x_active_mask': None,
            'expert_shard_type': 0,
            'shared_expert_num': 0,
            'shared_expert_rank_num': shared_expert_rank_num,
            'num_experts': num_experts,
            'scales': None,
            'quant_mode': 2,
            'num_max_dispatch_tokens_per_rank': 1,
        }

    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(
            backend='hccl',
            rank=rank,
            world_size=world_size,
            init_method=f'tcp://{master_ip}:{port}'
        )
        ep_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        # 创建输入tensor
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        topk_idx = torch.tensor([
            [0, 8, 4, 1, 6, 12, 14, 17],
            [14, 10, 7, 3, 0, 12, 11, 17],
            [12, 0, 5, 11, 19, 4, 6, 18],
            [17, 3, 4, 10, 18, 0, 1, 2],
            [13, 16, 9, 10, 15, 6, 7, 14],
            [17, 15, 14, 8, 16, 18, 3, 12],
            [4, 12, 2, 17, 15, 3, 9, 10],
            [16, 7, 12, 9, 18, 3, 19, 17]
        ], dtype=torch.int32).npu()
        topk_weights = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + num_experts, h) if shared_expert_rank_num else (num_experts, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None
        elastic_info = None
        const_expert_alpha_1 = None
        const_expert_alpha_2 = None
        const_expert_v = None
        model = MOE_DISTRIBUTE_GRAPH_Model(ep_group)
        model = model.npu()
        npu_backend = torchair.get_npu_backend()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        output = model.forward(
            x, topk_idx, ep_hcomm_info, ep_world_size,
            rank, 0, shared_expert_rank_num, num_experts, scales,
            quant_mode, globalBS, topk_weights, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
            zero_expert_num, copy_expert_num, const_expert_num
        )
        torch.npu.synchronize()
        print(f'rank {rank} npu finished! \n')

        time.sleep(10)


    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"num_max_dispatch_tokens_per_rank={globalBS}")
        print(f"shared_expert_rank_num={shared_expert_rank_num}")
        print(f"num_experts={num_experts}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
        print(f"ep_world_size={ep_world_size}", flush=True)

        if shared_expert_rank_num > ep_world_size:
            print("shared_expert_rank_num不能大于ep_world_size")
            exit(0)

        if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
            print("ep_world_size必须是shared_expert_rank_num的整数倍")
            exit(0)

        if num_experts % moe_rank_num != 0:
            print("num_experts必须是moe_rank_num的整数倍")
            exit(0)

        p_list = []
        for rank in range(rank_per_dev):
            p = Process(target=run_npu_process, args=(rank,))
            p_list.append(p)
        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

        print("run npu success.")
    ```
