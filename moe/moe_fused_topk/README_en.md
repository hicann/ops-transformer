# MoeFusedTopk

English | [简体中文](./README.md)

## Product Support

|Product                                |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Description

- **Operator function**: In MoE computation, this operator applies the Sigmoid computation to the input x, sorts the computation results by group, and finally selects the top k experts based on the grouped sorting results.
- **Formula**:

  Apply the sigmoid computation to the input:
  $$
  sigmoidRes=sigmoid(x)
  $$
  Add addNum:
  $$
  normOut = sigmoidRes + addNum
  $$
  Group the computation results by groupNum, sort the groups by the sum of the topN values in each group, and take the first groupTopk groups:
  $$
  groupOut, groupId = TopK(ReduceSum(TopK(Split(normOut, groupCount), k=topN, dim=-1), dim=-1),k=groupTopk)
  $$
  Set the normOut of the unselected groups to 0 (the selected groups remain unchanged), and then apply TopK to all experts to obtain the indices result:
  $$
  normOut' = \begin{cases}
  normOut, & group \in groupId \\
  0, & group \notin groupId
  \end{cases}
  $$
  $$
  normY,indices=TopK(normOut',k=k)
  $$
  > **Note**: The elements of unselected groups participate in the final TopK with the value 0 instead of being excluded. Therefore, when `addNum` makes `normOut` negative, the 0 of an unselected group may be ranked before the negative value of a selected group and thus be selected into `indices`.
  Select y from sigmoidRes based on indices:
  $$
  y = gather(sigmoidRes, indices)
  $$
  If isNorm is true, compute y using the input scale parameter to obtain the result of y:
  $$
  y = y / (ReduceSum(y, dim=-1))*scale
  $$
  If enableExpertMapping is true, map the physical experts in indices to logical experts based on the input mappingNum and mappingTable to obtain the output indices.

## Parameter Description

  <table style="table-layout: auto; width: 100%">
    <thead>
      <tr>
        <th style="white-space: nowrap">Parameter Name</th>
        <th style="white-space: nowrap">Input/Output/Attribute</th>
        <th style="white-space: nowrap">Description</th>
        <th style="white-space: nowrap">Data Type</th>
        <th style="white-space: nowrap">Data Format</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>Scores of each token corresponding to each expert, corresponding to `x` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>addNum</td>
        <td>Input</td>
        <td>Bias value computed with the input x, corresponding to `addNum` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>mappingNum</td>
        <td>Input</td>
        <td>Not used when `enableExpertMapping` is false. Number of logical experts to which each physical expert is actually mapped.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>mappingTable</td>
        <td>Input</td>
        <td>Not used when `enableExpertMapping` is false. Mapping table of each physical expert/logical expert.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>groupNum</td>
        <td>Attribute</td>
        <td>Number of groups.</td>
        <td>UINT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupTopk</td>
        <td>Attribute</td>
        <td>Number of selected groups.</td>
        <td>UINT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>topN</td>
        <td>Attribute</td>
        <td>Number of experts selected in each group for summation.</td>
        <td>UINT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>topK</td>
        <td>Attribute</td>
        <td>Number of experts finally selected.</td>
        <td>UINT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>activateType</td>
        <td>Attribute</td>
        <td>Activation type. Currently, only 0 (ACTIVATION_SIGMOID) is supported.</td>
        <td>UINT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>isNorm</td>
        <td>Attribute</td>
        <td>Whether to normalize the output.</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scale</td>
        <td>Attribute</td>
        <td>Coefficient multiplied after normalization.</td>
        <td>FLOAT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>enableExpertMapping</td>
        <td>Attribute</td>
        <td>Whether to enable the mapping from physical experts to logical experts.</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>Output</td>
        <td>Outputs the topK score of each token, corresponding to `y` in the formula.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>indices</td>
        <td>Output</td>
        <td>Mapping between the topK experts and tokens, corresponding to `indices` in the formula.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

- The data types of x and addNum must be the same.
- expertNum must be an integer multiple of groupNum.
- groupTopk is less than or equal to groupNum.
- maxMappingNum is less than or equal to 128.
- TopK is less than or equal to expertNum.
- TopN is less than or equal to expertNum / groupNum.
- expertNum is less than or equal to 1024.
- groupNum is less than or equal to 256.

## Calling Instructions

| Calling Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_moe_fused_topk](examples/test_aclnn_moe_fused_topk.cpp) | Call the MoeFusedTopk operator through the [aclnnMoeFusedTopk](docs/aclnnMoeFusedTopk_en.md) API. |

## Test Description

ACLNN regression test cases are stored in [ttk_aclnn_moe_fused_topk.csv](tests/st/arch35/ttk_aclnn_moe_fused_topk.csv), using [golden.py](tests/assets/golden.py). After installing the operator package for the corresponding product and configuring the CANN environment, run the following commands in the TTK repository directory (`OP_DIR` points to the directory of this operator):

```bash
python3 -m ttk aclnn -i "$OP_DIR/tests/st/arch35/ttk_aclnn_moe_fused_topk.csv" \
  --plugin "$OP_DIR/tests/assets/" --compare stat_rel_err --run 3 --dl 1
python3 -m ttk aclnn -i "$OP_DIR/tests/st/arch35/ttk_aclnn_moe_fused_topk.csv" \
  --plugin "$OP_DIR/tests/assets/" --ti 0-20 --compare binary --run 3 --dl 1
```

- The first 21 cases cover mapping number boundaries under FP32, FP16, and BF16, including the INT32 minimum value, negative numbers, zero, valid numbers, numbers exceeding the mapping table width, and the INT32 maximum value. Golden generates distinguishable expert scores and mapping tables for these cases. When the mapping number is less than or equal to zero, the first column is used. When the number exceeds the table width, the modulo of the table width is used. The second command compares outputs element by element for exact equality.
- The last 24 cases cover the output layouts of `y` and `indices`: the 14 valid layouts should pass the precision and determinism checks, and the 10 element-overlapping layouts should cause `aclnnMoeFusedTopkGetWorkspaceSize` to return `ACLNN_ERR_PARAM_INVALID` (161002) and should not enter the second-phase execution. The `remark.expected_acl_status` field of each case records the expected return code. TTK does not automatically use this field as an assertion. Cases expected to be rejected are displayed as N/A in the native report, so you need to check the return code. The `output_layouts` field in the API unit tests also asserts the return codes of these layouts.
