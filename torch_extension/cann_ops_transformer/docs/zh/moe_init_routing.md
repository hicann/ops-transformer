# moe\_init\_routing

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>                                      |    √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>  |    ×     |
|<term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|<term>Atlas 推理系列产品</term>    |     ×    |
|<term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- API功能：MoE的routing计算，根据torch_npu-npu_moe_gating_top_k_softmax的计算结果做routing处理，支持非量化、静态量化和动态量化模式。相较于`npu\_moe\_init\_routing\_v2`，新增可选输入`topk\_weight`和可选输出`expanded\_topk\_weight`，支持对topkWeight按排序后的索引进行重排，使得`expanded\_topk\_weight`与`expanded\_x`一一对应。`topk\_weight`与`expanded\_topk\_weight`必须同时传入或同时不传入。

- 计算公式：

  1.对输入expertIdx做排序，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx：

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

  2.以sortedRowIdx做位置映射得出expandedRowIdxOut：
    - rowIdxType等于1时, 输出scatter索引

      $$
      expandedRowIdxOut[i]=sortedRowIdx[i]
      $$

    - rowIdxType等于0时, 输出gather索引

      $$
      expandedRowIdxOut[sortedRowIdx[i]]=i
      $$

  3.对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensCountOrCumsumOutOptional：

    $$
    expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
    $$

  4.如果quantMode不等于-1, 计算quant结果：
     - 静态quant

     $$
     quantResult=round((x∗scaleOptional)+offsetOptional)
     $$

    - 动态quant：
        - 若不输入scale：

            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - 若输入scale:

            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - 当quantMode为13时，动态量化使用对称量化范围[-8, 7]，scale计算中的分母为7，量化结果沿H维每两个INT4值打包为1个字节。

  5.若活跃的expert范围为全专家范围时，按照Scatter索引搬运token；反之按照Gather索引搬运token。在dropPadMode为1时将每个专家需要处理的Token个数对齐为expertCapacity个，超过expertCapacity个的Token会被Drop，不足的会用0填充。得出expandedXOut：
    - 非量化场景
      - 按照Scatter索引搬运

      $$
      expandedXOut[i]=x[scatterRowIdx[i] // K]
      $$

      - 按照Gather索引搬运

      $$
      expandedXOut[gatherRowIdx[i]]=x[i // K]
      $$

    - 量化场景
      - 按照Scatter索引搬运

      $$
      expandedXOut[i]=quantResult[scatterRowIdx[i] // K]
      $$

      - 按照Gather索引搬运

      $$
      expandedXOut[gatherRowIdx[i]]=quantResult[i // K]
      $$

  6.若输入topkWeight，按照排序后的索引对topkWeight进行重排，使得expandedTopkWeightOut与expandedXOut一一对应：
    - Dropless场景（dropPadMode=0）

      $$
      expandedTopkWeightOut[i] = topkWeight[sortedRowIdx[i]], \quad 0 \le i < effectiveNum
      $$

      其中effectiveNum为：

      $$
      effectiveNum = \min(activeNum, availableIdxNum)
      $$

    - DropPad场景（dropPadMode=1）

      $$
      expandedTopkWeightOut[dstIdx] = topkWeight[globalSortIdx], \quad dstIdx \ne -1
      $$

      其中globalSortIdx为expertIdx在flatten布局中的位置（即globalSortIdx = rowIdx \* K + colIdx，对应topkWeight[rowIdx, colIdx]），dstIdx为expandedRowIdxOut中对应位置的目标索引，未被分配的专家容量位置expandedTopkWeightOut填充为0。

  7.expandedRowIdxOut的有效元素数量availableIdxNum，计算方式为expertIdx中activeExpertRangeOptional范围内的元素的个数

    $$
    availableIdxNum = |\{x\in expertIdx| expert\_start \le x<expert\_end \ \}|
    $$

## 函数原型

```python
def moe_init_routing(
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    *,
    scale: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
    topk_weight: Optional[torch.Tensor] = None,
    active_num: Union[int, torch.SymInt] = -1,
    expert_capacity: int = -1,
    expert_num: int = -1,
    drop_pad_mode: int = 0,
    expert_tokens_num_type: int = 0,
    expert_tokens_num_flag: bool = False,
    quant_mode: int = -1,
    active_expert_range: Optional[List[int]] = None,
    row_idx_type: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

## 参数说明

- **x** (`Tensor`)：必选参数，表示MoE的输入即token特征输入，要求为2维张量，shape为(NUM\_ROWS, H)。数据类型支持`float16`、`bfloat16`、`float32`、`int8`，数据格式要求为$ND$。
- **expert\_idx** (`Tensor`)：必选参数，表示torch_npu-npu_moe_gating_top_k_softmax输出每一行特征对应的K个处理专家，要求是2维张量，shape为(NUM\_ROWS, K)，且专家id不能超过专家数。数据类型支持`int32`，数据格式要求为$ND$。
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **scale** (`Tensor`)：可选参数，默认为None，用于计算量化结果的参数。数据类型支持`float32`，数据格式要求为$ND$。如果不输入表示计算时不使用`scale`，且输出`expanded\_scale`中的值无意义。
    - 非量化场景下，如果输入则要求为1维张量，shape为(NUM\_ROWS,)。
    - 静态量化场景必须输入，输入要求为1D的Tensor，shape为(1,)
    - 动态量化场景下，如果输入则要求为2维张量，shape为(expert\_end-expert\_start, H)或(1, H)。
    - quantMode为1的INT8动态量化场景下为可选输入，如果输入则要求为2D的Tensor，shape为(expert\_end-expert\_start, H)；quantMode为13的INT4动态量化场景下为可选输入，如果输入则要求shape为(1, H)，表示按H维广播的smooth scale。
    - MXFP8量化场景下（quantMode为2、3）不输入。

- **offset** (`Tensor`)：可选参数，默认为None，用于计算量化结果的偏移值。数据类型支持`float32`，数据格式要求为$ND$。
    - 在非量化场景下不输入。
    - 静态量化场景必须输入，输入要求为1维张量，shape为(1,)
    - 动态量化、MXFP8量化场景下不输入。

- **topk\_weight** (`Tensor`)：可选参数，默认为None，表示topk专家的路由权重，用于按排序索引重排以与`expanded\_x`一一对应。
    - 可选输入，不输入表示不输出`expanded\_topk\_weight`。
    - 输入shape为(NUM\_ROWS, K)，数据类型仅支持`float32`，数据格式要求为$ND$。
    - 与`expanded\_topk\_weight`联动：必须同时传入或同时不传入。

- **active\_num** (`Union[int, torch.SymInt]`)：可选参数，默认值为-1，表示总的最大处理row数，输出`expanded\_x`只有这么多行是有效的，入参校验需大于等于0，0表示Dropless场景，大于0时表示Active场景，约束所有专家共同处理tokens总量。支持`int`或`torch.SymInt`类型，图模式下传入`torch.SymInt`（如`x.size(0) * k`）可避免被固化为常量，实现动态shape编译缓存复用。eager模式下传入`int`即可，上层调用脚本无需感知类型差异，同一份代码在两种模式下均可正常运行。
- **expert\_capacity** (`int`)：可选参数，默认值为-1，表示每个专家能够处理的tokens数，入参校验大于0小于NUM\_ROWS。
- **expert\_num** (`int`)：可选参数，默认值为-1，表示专家数。`expert\_tokens\_num\_type`为key\_value模式时，取值范围为[0, 5120]；其他模式取值范围为[0, 10240]。
- **drop\_pad\_mode** (`int`)：可选参数，默认值为0，表示是否为drop\_pad场景。0表示Dropless场景，该场景下不校验`expert\_capacity`。1表示Drop\_Pad场景。
- **expert\_tokens\_num\_type** (`int`)：可选参数，默认值为0，表示直方图的不同模式。取值为0、1和2。0表示cumsum模式；1表示count模式；2表示key\_value模式。
- **expert\_tokens\_num\_flag** (`bool`)：可选参数，默认值为False，取值为False和True，表示是否输出`expert\_token\_cumsum\_or\_count`。
- **quant\_mode** (`int`)：可选参数，默认值为-1，表示量化模式，支持取值为-1、0、1、2、3、4、5、6、7、8、9、11、12、13、14、15、16、17（不同产品支持情况有差异，见约束说明）。
    - -1：表示非量化场景；
    - 0：表示静态量化场景；
    - 1：表示动态量化场景，`expanded\_x`量化到INT8；
    - 2：表示MXFP8量化场景，`expanded\_x`量化到float8\_e5m2；
    - 3：表示MXFP8量化场景，`expanded\_x`量化到float8\_e4m3fn；
    - 4：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale），`expanded\_x`量化到float8\_e5m2；
    - 5：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale），`expanded\_x`量化到float8\_e4m3fn；
    - 6：表示HIF8直转量化场景，`expanded\_x`量化到hifloat8；
    - 7：表示HIF8 PERTENSOR量化场景；
    - 8：表示HIF8 PERTOKEN量化场景；
    - 9：表示MXFP4量化场景，`expanded\_x`量化到float4\_e2m1；
    - 11：表示FP8 PerBlock量化场景（BlockSize=128），`expanded\_x`量化到float8\_e5m2；
    - 12：表示FP8 PerBlock量化场景（BlockSize=128），`expanded\_x`量化到float8\_e4m3fn；
    - 13：表示INT4动态量化场景，`expanded\_x`量化到INT4；
    - 14：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale+Amax），`expanded\_x`量化到float8\_e5m2；
    - 15：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale+Amax），`expanded\_x`量化到float8\_e4m3fn；
    - 16：表示MXFP8 RoundScale+Amax量化场景，`expanded\_x`量化到float8\_e5m2；
    - 17：表示MXFP8 RoundScale+Amax量化场景，`expanded\_x`量化到float8\_e4m3fn。
- **active\_expert\_range** (`List[int]`)：可选参数，默认为None，表示活跃expert的范围。长度为2，数组内的值为[expert\_start, expert\_end]，左闭右开，要求值大于等于0，并且expert\_end不大于`expert\_num`。Drop/Pad场景下，expert\_start等于0, expert\_end等于`expert\_num`。传入None时，视为活跃的expert范围在0到`expert\_num`之间。
- **row\_idx\_type** (`int`)：可选参数，默认为0，表示输出`expanded\_row\_idx`使用的索引类型，支持取值0和1。0表示gather类型的索引；1表示scatter类型的索引。

## 返回值说明

- **expanded\_x** (`Tensor`)：根据`expert\_idx`进行扩展过的特征。Dropless场景shape为[NUM\_ROWS \* K, H]。Active场景shape为[min(activeNum, NUM\_ROWS \* K), H]。Drop/Pad场景下shape为[expertNum \* expertCapacity, H]。非量化场景下数据类型同`x`；量化场景下数据类型根据quantMode确定（详见参数说明中quant\_mode）。数据格式要求为$ND$。量化场景下，当`x`的数据类型为`int8`时，输出值无意义。
- **expanded\_row\_idx** (`Tensor`)：`expanded\_x`和`x`的映射关系，shape与`expanded\_x`第一维一致，数据类型支持`int32`，数据格式要求为$ND$。前availableIdxNum个元素为有效数据，其余无效数据由`row\_idx\_type`决定：当`row\_idx\_type`为0时，无效数据由-1填充；当`row\_idx\_type`为1时，无效数据未初始化。
- **expert\_token\_cumsum\_or\_count** (`Tensor`)：表示输出每个专家处理的token数量的统计结果或累加值。数据类型支持`int64`，数据格式要求为$ND$。
    - 在`expert\_tokens\_num\_type`为0时，表示`active\_expert\_range`范围内expert在排序后处理token总数的前缀和。
    - 在`expert\_tokens\_num\_type`为1的场景下，shape为[expert\_end-expert\_start]。
    - 在`expert\_tokens\_num\_type`为2的场景下，shape为[expert\_num, 2]，表示token总数为非0的expert及其处理token的总数。

- **expanded\_scale** (`Tensor`)：输出不同量化过程中scale的中间值。数据类型支持`float32`或`float8\_e8m0`，数据格式要求为$ND$。
    - 非量化场景下，当`scale`输入时，shape为[NUM\_ROWS\*K, 1]，前availableIdxNum个元素为有效数据，输出`float32`类型。当输入x数据类型为`float8_e5m2`、`float8_e4m3fn`或`float4_e2m1`时，如果`scale`输入，则shape为[NUM\_ROWS\*K, CeilDiv(H, 64), 2]，输出`float8_e8m0`类型。
    - 动态量化场景下，当`scale`输入时，前availableIdxNum个元素为有效数据。
    - 静态量化场景下、HIF8直转量化场景下、HIF8 PERTENSOR量化场景下，输出为空tensor。
    - HIF8 PERTOKEN量化场景下，shape为[NUM\_ROWS\*K, 1]，输出`float32`类型。
    - MXFP8量化场景下（quantMode为2、3、16、17），输出`float8\_e8m0`类型，Shape为[NUM\_ROWS\*K, M]，其中M=CeilAlign(CeilDiv(H, 32), 2)，前availableIdxNum行为有效数据。
    - MXFP4量化场景下，输出`float8\_e8m0`类型，Shape为[NUM\_ROWS\*K, M, 2]，其中M=CeilDiv(H, 64)，前availableIdxNum行为有效数据。
    - FP8 PerGroup量化场景下（quantMode为4、5、14、15），输出`float32`类型，Shape为[NUM\_ROWS\*K, CeilDiv(H, 128)]，前availableIdxNum行为有效数据。
    - FP8 PerBlock量化场景下（quantMode为11、12），输出`float32`类型，Shape为[NUM\_ROWS\*K, CeilDiv(H, 256), 2]，前availableIdxNum行为有效数据。

- **expanded\_topk\_weight** (`Tensor`)：按排序索引重排后的路由权重，与`expanded\_x`一一对应。数据类型仅支持`float32`，数据格式要求为$ND$。
    - 可选输出，`topk\_weight`输入时必须同时输出；`topk\_weight`未输入时输出为空tensor（shape为(0,)）。
    - Dropless场景下shape为[min(activeNum, NUM\_ROWS\*K), 1]，前effectiveNum个元素为有效数据，其余未初始化，effectiveNum=min(activeNum, availableIdxNum)。
    - DropPad场景下shape为[expertNum\*expertCapacity, 1]，已分配容量的位置为有效数据，未分配容量的位置填充为0。

## 约束说明

1. 该接口支持推理场景下使用。
2. 该接口支持图模式。
3. `x`和`expert\_idx`必须是2D张量。
4. `topk\_weight`和`expanded\_topk\_weight`必须同时传入或同时不传入（联动约束）。
5. `topk\_weight`仅支持`float32` dtype，shape必须为(NUM\_ROWS, K)。
6. `topk\_weight`不受`quant\_mode`影响，`expanded\_topk\_weight`数据类型始终为`float32`。
7. `topk\_weight`/`expanded\_topk\_weight`仅<term>Ascend 950PR/Ascend 950DT</term>支持，其他产品不支持该参数。
8. `expert\_num`必须大于0。
9. `drop\_pad\_mode=1`时，`expert\_capacity`必须大于0，且具有如下额外约束：`row\_idx\_type`仅支持取值为0（gather索引）；`active\_expert\_range`必须为[0, expert\_num]；`quant\_mode`仅支持-1（非量化）。
10. `expert\_tokens\_num\_type`仅支持0、1、2。
11. `row\_idx\_type`仅支持0或1。
12. quantMode支持情况：
    - <term>Ascend 950PR/Ascend 950DT</term>：支持-1、0、1、2、3、4、5、6、7、8、9、11、12、13、14、15、16、17。
13. <term>Ascend 950PR/Ascend 950DT</term>仅支持如下参数值：
    - activeNum仅支持值等于NUM\_ROWS\*K。
    - expertCapacity在Dropless场景下仅校验其值，不使用该参数；在DropPad场景下取值范围为(0, NUM\_ROWS]。
    - expertTokensNumFlag仅支持取值为true。
14. <term>Ascend 950PR/Ascend 950DT</term>支持quantMode为13的INT4动态量化场景，需同时满足：x数据类型为float32或bfloat16，expanded\_x数据类型为int4；H为偶数；scale不输入或输入shape为(1, H)；offset不输入。
15. 空tensor处理：当输入的x首个维度的值为0时，DropPadMode必须为0，expanded\_x、expanded\_row\_idx和expanded\_scale为空tensor，expert\_token\_cumsum\_or\_count返回全0的tensor。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import moe_init_routing

    n = 3
    h = 2
    k = 4
    expert_num = 256
    expert_capacity = -1
    drop_pad_mode = 0
    expert_tokens_num_type = 1
    expert_tokens_num_flag = True
    quant_mode = -1
    active_expert_range = [0, 4]
    row_idx_type = 1

    x = torch.randn((n, h), dtype=torch.float32).npu()
    expert_idx = torch.randint(0, expert_num, (n, k), dtype=torch.int32).npu()
    scale = torch.randn((n,), dtype=torch.float32).npu()

    # 不传入topk_weight，expanded_topk_weight为空tensor
    expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale, expanded_topk_weight = \
        moe_init_routing(x, expert_idx, scale=scale,
                         active_num=-1, expert_capacity=expert_capacity,
                         expert_num=expert_num, drop_pad_mode=drop_pad_mode,
                         expert_tokens_num_type=expert_tokens_num_type,
                         expert_tokens_num_flag=expert_tokens_num_flag,
                         quant_mode=quant_mode, active_expert_range=active_expert_range,
                         row_idx_type=row_idx_type)

    # 传入topk_weight，expanded_topk_weight为有效数据
    topk_weight = torch.randn((n, k), dtype=torch.float32).npu()
    expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale, expanded_topk_weight = \
        moe_init_routing(x, expert_idx, topk_weight=topk_weight, scale=scale,
                         active_num=-1, expert_capacity=expert_capacity,
                         expert_num=expert_num, drop_pad_mode=drop_pad_mode,
                         expert_tokens_num_type=expert_tokens_num_type,
                         expert_tokens_num_flag=expert_tokens_num_flag,
                         quant_mode=quant_mode, active_expert_range=active_expert_range,
                         row_idx_type=row_idx_type)
    ```

    - 图模式调用（静态量化场景，quant\_mode=0）

    ```python
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    from cann_ops_transformer.ops import moe_init_routing

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class MoeInitRoutingModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, expert_idx, *, topk_weight=None, scale=None, offset=None,
                    active_num=-1, expert_capacity=-1, expert_num=-1, drop_pad_mode=0,
                    expert_tokens_num_type=0, expert_tokens_num_flag=False, quant_mode=-1,
                    active_expert_range=None, row_idx_type=0):
            return moe_init_routing(x, expert_idx, topk_weight=topk_weight,
                                    scale=scale, offset=offset, active_num=active_num,
                                    expert_capacity=expert_capacity, expert_num=expert_num,
                                    drop_pad_mode=drop_pad_mode,
                                    expert_tokens_num_type=expert_tokens_num_type,
                                    expert_tokens_num_flag=expert_tokens_num_flag,
                                    quant_mode=quant_mode,
                                    active_expert_range=active_expert_range,
                                    row_idx_type=row_idx_type)

    def main():
        n = 3
        h = 2
        k = 4
        expert_num = 256
        active_expert_range = [0, 4]

        x = torch.randn((n, h), dtype=torch.float32).npu()
        expert_idx = torch.randint(0, expert_num, (n, k), dtype=torch.int32).npu()
        topk_weight = torch.randn((n, k), dtype=torch.float32).npu()
        scale = torch.randn((1,), dtype=torch.float32).npu()
        offset = torch.randn((1,), dtype=torch.float32).npu()

        model = MoeInitRoutingModel().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale, expanded_topk_weight = \
            model(x, expert_idx, topk_weight=topk_weight, scale=scale, offset=offset,
                  active_num=-1, expert_capacity=-1, expert_num=expert_num,
                  drop_pad_mode=0, expert_tokens_num_type=1,
                  expert_tokens_num_flag=True, quant_mode=0,
                  active_expert_range=active_expert_range, row_idx_type=1)

    if __name__ == '__main__':
        main()
    ```
