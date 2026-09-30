# Data Structures

This section describes the basic data structures required for calling CANN operator APIs. You may directly use these structures without delving into their internal implementations.

Note that the basic data structures can be created by using the common APIs such as **aclCreateTensor** in "Common APIs" in [Operator Library API Reference](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/aolapi/operatorlist_00001.html)

- **aclTensor**

  A framework-defined structure used to manage and store tensor data (such as vectors and matrices). You can create this object by using the **aclCreateTensor** API.

  ```text
  typedef struct aclTensor aclTensor
  ```

- **aclScalar**

  A framework-defined structure used to manage and store scalar data (a single value). You can create this object by using the **aclCreateScalar** API.

  ```text
  typedef struct aclScalar aclScalar
  ```

- **aclIntArray**

  A framework-defined structure used to manage and store an array of integer data. You can create this object by using the **aclCreateIntArray** API.

  ```text
  typedef struct aclIntArray aclIntArray
  ```

- **aclFloatArray**

  A framework-defined structure used to manage and store an array of float32 data. You can create this object by using the **aclCreateFloatArray** API.

  ```text
  typedef struct aclFloatArray aclFloatArray
  ```

- **aclBoolArray**

  A framework-defined structure used to manage and store an array of Boolean data. You can create this object by using the **aclCreateBoolArray** API.
    
  ```text
  typedef struct aclBoolArray aclBoolArray
  ```
    
- **aclTensorList**

  A framework-defined structure used to manage and store an array of multiple tensors. You can create this object by using the **aclCreateTensorList** API.
    
  ```text
  typedef struct aclTensorList aclTensorList
  ```
    
- **aclScalarList**

  A framework-defined structure used to manage and store an array of scalar data. You can create this object by using the **aclCreateScalarList** API.

  ```text
  typedef struct aclScalarList aclScalarList
  ```

- **aclOpExecutor**

  A framework-defined executor data structure, which is a container for operator execution.

  Generally, when the first-phase API **aclxxXxxGetWorkspaceSize** is called, the framework automatically creates an **aclOpExecutor** object. When the second-phase API **aclxxXxx** is called, the object is automatically released.

  ```text
  typedef struct aclOpExecutor aclOpExecutor
  ```

- **aclrtStream**

  A framework-defined stream processing data structure, which is used to manage and maintain the execution sequence of asynchronous operations.
    
  ```text
  typedef void *aclrtStream
  ```
