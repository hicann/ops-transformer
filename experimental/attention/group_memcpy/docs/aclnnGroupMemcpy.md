# aclnnGroupMemcpy

## 产品支持情况

<!-- npu="910b" id1 -->
- <term>Atlas A2系列产品</term>：配置支持 `ascend910b`；本次未在产品上完成编译和运行验证。
<!-- end id1 -->

其他产品未配置。产品配置来自 [OpDef](../op_host/group_memcpy_def.cpp)，不能据此推定已通过上板测试。

## 功能说明

成组拷贝分散的连续 Tensor。设列表长度为 `N`、每个 Tensor 的有效字节数为 `W`，则：

~~~text
对 i = 0 ... N-1:
    对 b = 0 ... W-1:
        dst[i].byte[b] = src[i].byte[b]
~~~

此处 byte 从 Tensor 的视图起点开始，不是底层 storage 起点。每对 shape 相同，所有 dtype 和元素数相同；不同组可以有不同 shape，但元素数相等。该接口不做数值运算，NaN 载荷、正负零等位模式保持不变。

先调用 `aclnnGroupMemcpyGetWorkspaceSize` 获取 workspace 大小和执行器，再调用 `aclnnGroupMemcpy` 提交到指定 stream。目标内存由调用方预分配；`const aclTensorList*` 约束描述符接口，不表示其指向的数据不会被写入。

## 函数原型

~~~cpp
aclnnStatus aclnnGroupMemcpyGetWorkspaceSize(
    const aclTensorList* src,
    const aclTensorList* dst,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
~~~

~~~cpp
aclnnStatus aclnnGroupMemcpy(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
~~~

头文件：[aclnn_group_memcpy.h](../op_host/op_api/aclnn_group_memcpy.h)。应链接本次构建安装的 `libcust_opapi.so` 或对应 `libopapi_transformer.so`，由实际打包方式决定。

## aclnnGroupMemcpyGetWorkspaceSize

<table style="table-layout: fixed; width: 1500px">
<thead><tr><th>参数名</th><th>输入/输出</th><th>描述</th><th>使用说明</th><th>数据类型</th><th>数据格式</th><th>维度(shape)</th><th>非连续Tensor</th></tr></thead>
<tbody>
<tr><td>src（const aclTensorList*）</td><td>输入</td><td>N 个源 Tensor，对应 src[i]</td><td>非空指针及非空列表；允许零元素 Tensor；所有元素数及 dtype 相同</td><td>INT8、UINT8、INT16、INT32、INT64、FLOAT16、BFLOAT16、FLOAT、BOOL</td><td>ND</td><td>各元素可为标量或多维，受 aclTensor 本身限制</td><td>×</td></tr>
<tr><td>dst（const aclTensorList*）</td><td>输出</td><td>N 个预分配目标 Tensor，对应 dst[i]</td><td>列表长度与 src 一致；每对 shape 相同；所有 dtype 与 src 一致；允许零元素 Tensor</td><td>与 src 相同</td><td>ND</td><td>dst[i].shape = src[i].shape</td><td>×</td></tr>
<tr><td>workspaceSize（uint64_t*）</td><td>输出</td><td>执行需要的临时内存字节数</td><td>非空；以实际返回值分配，不能假设恒为零</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>executor（aclOpExecutor**）</td><td>输出</td><td>包含本次搬运流程的执行器</td><td>非空；成功后传入第二段；失败时不调用第二段</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
</tbody>
</table>

`src` 和 `dst` 是 CPU 侧描述符列表，元素描述 Device Tensor，不是把原来的设备裸指针数组直接强转为 `aclTensorList*`。

**返回值：**

| 状态 | 触发条件 |
| --- | --- |
| ACLNN_SUCCESS | 参数有效，成功创建执行器及搬运任务；零字节时创建空任务执行器 |
| ACLNN_ERR_PARAM_NULLPTR | src、dst、workspaceSize、executor 或列表元素描述符为空 |
| ACLNN_ERR_PARAM_INVALID | 空列表、长度不一致或超过 INT32_MAX；不支持的 dtype/格式；dtype、配对 shape 或各组字节数不一致；不连续；非法负维度或字节数溢出 |
| ACLNN_ERR_INNER_CREATE_EXECUTOR | 创建执行器失败 |
| 框架返回的其他状态 | 任务构造等内部步骤失败时原样返回 |

当 `workspaceSize` 和 `executor` 输出指针均有效时，参数校验失败会将其分别置零、置空。C 层不读取设备地址内容，也不检测各分配的设备、实际容量或地址重叠。

## aclnnGroupMemcpy

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| workspace（void*） | 输入 | Device 临时内存；workspaceSize 非零时不能为 nullptr |
| workspaceSize（uint64_t） | 输入 | 提供的 workspace 字节数，须满足第一段返回的要求 |
| executor（aclOpExecutor*） | 输入 | 第一段成功返回的执行器，不能为空 |
| stream（aclrtStream） | 输入 | 当前设备的有效执行流；有效性由 ACL 运行时检查 |

返回 `ACLNN_SUCCESS` 表示成功提交，执行完成需通过 stream 同步或事件依赖确认。空 executor 或非零 workspaceSize 对应空 workspace 返回 `ACLNN_ERR_PARAM_NULLPTR`；其他错误由 `CommonOpExecutorRun` 返回。

## 约束说明

1. `1 ≤ N ≤ INT32_MAX`，`0 ≤ W ≤ INT64_MAX - 64`；可分配的实际设备内存及框架资源通常给出更小上限。所有维度须为已知非负值。
2. 每组完整拷贝，无公开 `bytesPerCopy` 参数；要拷贝缓存的一部分，应先创建覆盖那一段的连续 Tensor 视图。
3. 必须使用同一设备上的有效 Device 地址，并保证视图范围位于分配内。允许合法非零 storage offset。
4. 任何目标写入范围不能与任意源读取范围或其他目标写入范围相交。重复源允许；源和目标为同一段内存也属于禁止的重叠。调用方负责这些条件。
5. 不允许非连续 Tensor、私有存储格式或 dtype 转换。空列表非法；非空列表中的各组均为零字节时不发射 Kernel。
6. 在 stream 完成前保持输入、输出和 workspace 的 Device 内存有效。描述符按 CANN TensorList 所有权规则管理，不能重复释放成员。
7. 不支持自动求导。本次交付提供 eager 调用和 Meta 注册，未提供 torchair 图转换器，也未验证图编译。

无重叠且遵守 stream 依赖时，结果为确定的按字节复制。所有测试使用字节完全相等判定，不使用浮点容差。

## 调用示例

完整可运行的两段式调用、资源释放和逐字节验证见 [C++ 测试](../tests/aclnn/main.cpp)。构建、安装、Torch 调用与测试命令见 [README](../README.md) 及 [测试说明](../tests/README.md)。
