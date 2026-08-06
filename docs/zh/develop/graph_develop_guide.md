# 图模式适配指南

## 概述

自定义算子如需运行图模式，整体流程与算子开发指南（[AI Core算子开发指南](aicore_develop_guide.md)）一致。值得注意的是，**不需要aclnn适配**，只需做如下交付件适配。

```text
${op_name}                              # 替换为实际算子名的小写下划线形式
├── op_host                             # Host侧实现
│   └── ${op_name}_infershape.cpp       # InferShape实现，实现算子形状推导，在运行时推导输出shape
├── op_graph                            # 图融合相关实现
│   ├── CMakeLists.txt                  # op_graph侧cmakelist文件
│   ├── ${op_name}_graph_infer.cpp      # InferDataType文件，实现算子类型推导，在运行时推导输出dataType
└── └── ${op_name}_proto.h              # 算子原型定义，用于图优化和融合阶段识别算子
```

本文将以`AddExample`算子（假设为AI Core算子）入图为例，介绍入图交付件的实现，完整代码详见`examples`目录下`add_example`。

## Shape与DataType推导

图模式需要完成两个交付件```${op_name}_graph_infer.cpp``` ```${op_name}_infershape.cpp```

**交付件1：${op_name}_infershape.cpp**

InferShape函数的作用是根据输入的shape推导输出的shape。

示例如下，`AddExample`算子完整代码请参考`examples/add_example/op_host`下[add_example_infershape.cpp](../../../examples/add_example/op_host/add_example_infershape.cpp)。

```cpp
// AddExample算子逻辑是两个数相加，因此输出shape与输入shape一致
static ge::graphStatus InferShapeAddExample(gert::InferShapeContext* context)
{
    ....
    // 获取输入shape
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    // 获取输出shape
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    // 获取输入DimNum
    auto xShapeSize = xShape->GetDimNum();
    // 设置输出的DimNum
    yShape->SetDimNum(xShapeSize);
    // 依次将输入Dim值设置给输出
    for (size_t i = 0; i < xShapeSize; i++) {
        int64_t dim = xShape->GetDim(i);
        yShape->SetDim(i, dim);
    }
    ...
}
// inferShape注册
IMPL_OP_INFERSHAPE(AddExample).InferShape(InferShapeAddExample);
```

**交付件2：${op_name}_graph_infer.cpp**

InferDataType函数的作用是根据输入的DataType推导输出的DataType。

示例如下，`AddExample`算子完整代码请参考`examples/add_example/op_graph`下[add_example_graph_infer.cpp](../../../examples/add_example/op_graph/add_example_graph_infer.cpp)。

```cpp
// AddExample算子逻辑是两个数相加，因此输出dataType与输入dataType一致
static ge::graphStatus InferDataTypeAddExample(gert::InferDataTypeContext* context)
{
    ...
    // 获取输入的dataType
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    // 将输入dataType设置到输出
    context->SetOutputDataType(IDX_0, sizeDtype);
    ...
}

// 注册InferDataType
IMPL_OP(AddExample).InferDataType(InferDataTypeAddExample);
```

## 算子原型配置

图模式调用需要将算子原型注册到[Graph Engine](https://www.hiascend.com/cann/graph-engine)（简称GE）中，以便GE能够识别该类算子的输入、输出及属性信息。注册通过`REG_OP`接口完成，开发者需定义算子输入、输出张量类型及数量等基本信息。

常用张量/属性数据类型示例如下：

|张量类型|属性类型|示例|
|-----|------|-----|
|int64|/|DT_INT64|
|int32|/|DT_INT32|
|int16|/|DT_INT16|
|int8|/|DT_INT8|
|double|/|DT_DOUBLE|
|float32|/|DT_FLOAT|
|float16|/|DT_FLOAT16|
|bfloat16|/|DT_BF16|
|complex128|/|DT_COMPLEX128|
|complex64|/|DT_COMPLEX64|
|complex32|/|DT_COMPLEX32|
|/|int|Int|
|/|bool|Bool|
|/|string|String|
|/|float|Float|
|/|list|ListInt|

基本信息如下：

|输入/输出|关键字|示例|
|-----|------|-----|
|必选输入|INPUT|.INPUT(${name}, TensorType({input_dtype}))|
|可选输入|OPTIONAL_INPUT|.OPTIONAL_INPUT(${name}, TensorType({optional_input_dtype}))|
|必选属性|REQUIRED_ATTR|.REQUIRED_ATTR(${name}, ${dtype})|
|可选属性|ATTR|.ATTR(${name}, ${dtype}, ${default_value})|
|输出|OUTPUT|.OUTPUT(${name}, TensorType({output_dtype}))|

示例代码如下，展示了如何注册`AddExample`算子：

```cpp
REG_OP(AddExample)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(AddExample)
```

完整代码请参考`examples/add_example/op_graph`目录下[add_example_proto.h](../../../examples/add_example/op_graph/add_example_proto.h)。
