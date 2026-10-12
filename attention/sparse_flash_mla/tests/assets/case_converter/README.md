# sparse_flash_mla Excel 用例转换工具

本目录将本算子的参数 Excel 表转换为 TTK E2E 或 ACLNN Excel 用例表。
`convert.py` 负责工作簿读写与追加，`case_schema.py` 负责本算子的 tensor、属性和输出描述。
整个目录可独立复制使用，各算子之间没有导入或路径依赖。

## 运行条件

Python 3.9+ 和 `openpyxl>=3.1`。转换只依赖 Python 标准库和 openpyxl，不需要 PyTorch、pytest、CANN 或 NPU。
已有 Python 环境缺少依赖时可执行 `python3 -m pip install "openpyxl>=3.1"`。

下列命令从当前目录执行，也可以把 `convert.py` 换成其完整路径，从任意目录调用。
输入支持 `.xlsx/.xlsm`，输出也是 Excel。

## 三种输出方式

1. 输出到目录，保持输入文件名和选中的 sheet 名：

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 --output-dir output --mode e2e
```

2. 输出到指定 Excel 文件，按输入 sheet 名创建或追加：

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 -o cases.xlsx --mode aclnn
```

3. 输出到指定 Excel 文件的指定 sheet，按输入选择顺序合并追加：

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 -o cases.xlsx --output-sheet Cases --mode e2e
```

`-s/--sheet` 可指定多个名称，也可重复使用；带空格的名称请加引号。
未指定时读取第一张 sheet，重复或不存在的名称报错；名称不按逗号拆分。
`--mode` 缺省为 `e2e`，`--output-dir` 与 `-o/--output` 互斥。

目录输出和指定文件输出遇到同名 sheet 时，工具会先询问是否追加。
只接受 `yes`、`y`、`Y`；其他回答或输入结束取消整次写入，已经确认的其他 sheet 也不写入。
sheet 名比较不区分大小写，保留目标现有拼写。
显式提供 `--output-sheet` 表示已指定追加目标，不重复确认。

## 输入表格式

第一行非空行为表头，一行代表一个用例；忽略全空行。
支持两种来源：本算子的 pytest 参数表，以及已包含 `tensor_view_shapes/tensor_dtypes/attributes` 的 E2E 描述表。
输入 ACLNN 描述表会报错，避免重复附加输出 tensor。
list、tuple、字典使用 Python 字面量，例如 `[0, 16, 32]`、`(1, 16, 8, 128)`、`{'layout_q': 'TND'}`。
表达式使用字面量解析，不能调用 Python 函数。

主要参数为 `B/S1/S2/N1/N2/D`、`layout_q/layout_kv`、`q_type/ori_kv_type`，dtype 缺省与可选 API 属性见本目录 `case_schema.py`。
`testcase_name`、`Testcase_Name`、`Testcase_Prefix` 或 `case_name` 可作为用例名称；同一行的名称别名必须一致。

- 查询布局支持 `BSND/TND`；KV 布局支持 `BSND/TND/PA_BBND`。
- `template_run_mode`（也接受 `template_mode`）支持 `SWA/HCA/CSA/ORI_SPARSE/ORI_CMP_SPARSE`。
  ORI sparse 分支需要 `K1`，CMP sparse 分支需要 `K`；没有模板字段时按 `K/cmp_ratio` 推导。
- 可提供 `seqused_q`、`seqused_ori_kv`、`seqused_cmp_kv` 及各自 `cu_seqlens_*`。
  累积序列从 0 开始，包含 `B+1` 个值；缺省为每个 batch 的完整容量。
  指定非均匀 TND 容量时必须提供累积序列，`T1/T2/T3`（若指定）必须匹配末端。
- `block_size1/block_size2` 缺省 128；可提供 `block_num1/block_num2`，不足时拒绝转换。
- `ori_mask_mode/cmp_mask_mode`、`ori_win_left/ori_win_right`、`topk_value_mode`、
  `return_softmax_lse` 保留为 API 属性。`ori_topk_length/cmp_topk_length` 使用独立的 `*_values` 数据属性，避免与 tensor 入参重名；一个数值展开到所有 query/KV head，多个值须匹配该 tensor 大小，范围为 [0, K]。
- `q_datarange/ori_kv_datarange/cmp_kv_datarange` 使用两个端点，支持 `inf/-inf/nan`。
  缺少完整或随机序列的确定值时拒绝推测随机结果。

`is_enabled` 缺省 True；显式 False 保持 False，不根据历史测试结果修改启用状态。
`precision_tolerances/absolute_precision` 可显式覆盖转换默认值。
`remark/network_name/priority/soc_series/golden_api/device_ids` 及已有 batch 字段按表中值保留。
`attributes1`..`attributes9` 合并进 `attributes`；后续列覆盖之前的同名属性。
输出格式的部分字段只由对应 TTK 模式使用。

## 输出与追加合同

- E2E API：`torch.ops.cann_ops_transformer.sparse_flash_mla`，包含 18 个输入 tensor 槽位；可选 tensor 保持 None。
- ACLNN API：`aclnnSparseFlashMla`，在输入后添加真实输出形状和 dtype，并填写 `output_tensor_indexes`。
- 输入和输出不得为同一个文件；拒绝同文件的 symlink/hardlink 别名，目标文件不能是 symlink。
- 追加按目标表头名称对齐，保留旧列和旧行；缺少的新列追加到表头末尾。一个 sheet 不能混放 E2E 和 ACLNN。
- 写出前检查所有选中 sheet 和目标表；失败不修改目标文件。写入使用同目录临时文件，再原子替换。
- 现有目标必须是普通 TTK 用例表；不向受保护、合并单元格或 Excel structured table 的用例页追加。
  其他 sheet 的现有内容、公式和单元格样式随工作簿保留。
- 输入公式须有 Excel 保存的计算缓存；工具不计算公式。无缓存时请先在 Excel 中重新计算并保存。
- `.xlsm` 目录输出保留源 VBA；追加已有 `.xlsm` 保留目标 VBA。新建 `.xlsm` 需要 `.xlsm` 来源。
  转成 `.xlsx` 只输出用例表。
- 成功退出码为 0，输入错误、拒绝追加或写入失败为 1。

## 与 TTK 执行的关系

转换生成参数描述；实际执行沿用本算子 assets 中的输入、Golden 和 TestSpec。
参数表输出会包含现有输入适配器需要的属性。执行仍需对应版本 TTK、算子包和设备环境。

metadata 输入使用 `(-1,)` 延迟分配，由现有 `npu_preprocess` 在执行时生成；
不另建 metadata sheet，不在转换机器上调用 metadata 算子。输入的既有静态 metadata 描述保持。
关闭 `return_softmax_lse/return_value` 时，ACLNN 对应可选输出为 `(0,)`。
