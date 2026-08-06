# 两段式接口

基于单算子API执行方式调用算子API时，通常分为“两段式”，样式形如：

```cpp
aclnnStatus aclxxXxxGetWorkspaceSize(const aclTensor *src, ..., aclTensor *out, ..., uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclxxXxx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

必须先调用第一段接口aclxxXxxGetWorkspaceSize，用于计算本次API调用过程中需要多少workspace内存，获取到计算所需的workspaceSize后，按照workspaceSize申请NPU内存，然后调用第二段接口aclxxXxx执行计算。

其中“aclxx”表示算子接口前缀，如aclnn；而“Xxx”表示对应的算子类型，如Add算子。

> 说明：
>
> - workspace是指除输入/输出外，算子在NPU上完成计算所需要的临时内存，workspaceSize表示临时内存的大小。
> - 第二段接口aclxxXxx(...)不能重复调用，如下调用方式会出现异常：
>
>   ```cpp
>   aclxxXxxGetWorkspaceSize(...)
>   aclxxXxx(...)
>   aclxxXxx(...)
