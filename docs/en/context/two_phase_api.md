# Two-Phase API

An operator API that is called in single-operator API execution mode is usually divided into two phases, for example:

```Cpp
aclnnStatus aclxxXxxGetWorkspaceSize(const aclTensor *src, ..., aclTensor *out, ..., uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclxxXxx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

First, **aclxxXxxGetWorkspaceSize** is called to compute the required workspace size in the process, based on which NPU memory is allocated. Then, **aclxxXxx** is called to perform computation.

**aclxx** indicates the operator API prefix, for example, aclnn, while **Xxx** indicates the operator type, for example, Add.

> Note:
>
>- The workspace refers to the temporary memory required by the APIs to complete computation on the AI processor, in addition to the input and output memories.
>- The second-phase API **aclxxXxx(...)** cannot be called repeatedly. For example, the following API call will throw an exception:
    ```Cpp     
    aclxxXxxGetWorkspaceSize(...)  
    aclxxXxx(...)   
    aclxxXxx(...)
    ```
