# torch_extension Development Guidelines

This document defines the conventions for directory organization, naming, implementation of each layer, and documentation when you add or modify operator APIs in `cann_ops_transformer` (torch_extension). Read this document in full before you add a new operator, and refer to existing operators (such as `flash_attn`) as templates.

`cann_ops_transformer` uses PyTorch JIT (`torch.utils.cpp_extension.load`) to compile the C++ kernel wrapper just-in-time on the first call, bridges the PyTorch function interface to the aclnn interface of CANN, and supports the torchair graph mode through a GE Converter. A complete operator API usually consists of four parts: the Python frontend, the C++ backend, the torchair graph mode Converter, and the documentation.

## 1. Overview of Operator API Directory Organization Conventions

Developing a new operator API involves adding and modifying the following files (using the operator `${op_api}` as an example):

```
ops-transformer/
├── <category>/<op>/
│   ├── op_host/                                     # existing operator implementation code
│   ├── op_kernel/                                   # existing kernel code
│   ├── tests/                                       # existing tests
│   └── torch_extension/                             # torch_extension files
│       ├── __init__.py                              # exports the operator interface and the Converter
│       ├── ${op_api}.py                             # Python frontend
│       ├── graph_convert_${op_api}.py               # torchair graph mode Converter (optional)
│       └── csrc/
│           └── ${op_api}.cpp                        # C++ backend
├── torch_extension/
│   ├── setup.py                                     # automatically collects distributed operator files
│   ├── cann_ops_transformer/
│   │   ├── __init__.py                              # package root entry, dynamic discovery + namespace export
│   │   ├── op_builder/
│   │   │   ├── __init__.py                          # exports OpBuilder, get_as_library
│   │   │   └── builder.py                           # OpBuilder base class (supports the category parameter and deferred initialization)
│   │   ├── common/
│   │   │   ├── aclnn_common.h                       # ACLNN_CMD macro, type conversion, and other common capabilities
│   │   │   └── hccl_common.h                        # common capabilities for communication operators
│   │   ├── csrc/
│   │   │   └── extension.cpp                        # C++ stub
│   │   ├── ops/
│   │   │   └── __init__.py                          # automatically discovers operators (directory scan + entry point)
│   │   └── docs/                                    # development guidelines and more (API documentation is placed in the operator docs/)
```

The standard action checklist for adding a new operator API (using `mc2/mega_moe` as an example):

1. Implement the C++ kernel wrapper in `mc2/mega_moe/torch_extension/csrc/mega_moe.cpp`, and call `ACLNN_CMD` to invoke the aclnn interface;
2. Write the `OpBuilder` subclass in `mc2/mega_moe/torch_extension/mega_moe.py` (define `sources`/`schema`/`register_meta`), register the dispatcher implementation, and provide the public Python function;
3. Write the graph mode Converter in `mc2/mega_moe/torch_extension/graph_convert_mega_moe.py` (if graph mode support is required);
4. Define `__all__` in `mc2/mega_moe/torch_extension/__init__.py`, and export the operator interface and the Converter;
5. After the operator is imported, the automatic discovery mechanism in `ops/__init__.py` loads it automatically; manual registration is not required;
6. Add the operator documentation in `<category>/<op>/docs/torchapi_${op_api}.md`, and update `docs/zh/menu_torch_api.md` and `docs/zh/torch_api_list.md` accordingly.

> Place new files in the `<category>/<op>/torch_extension/` directory of the corresponding operator, and use `cann_ops_transformer` as the root of all import paths.

## 2. Naming Conventions

### 2.1 API Naming

An operator involves naming at multiple levels from schema registration to public export; these names must stay consistent, and each level serves its own purpose. **Public API interfaces and operator names never carry the `npu_` prefix**; they directly use the lowercase snake_case name that reflects the operator semantics (such as `flash_attn`):

| Level | Naming Convention | Example |
| --- | --- | --- |
| Library name (DEF domain) | Fixed to `cann_ops_transformer` | The Library instance created by `get_as_library()` |
| Import path | Import from `cann_ops_transformer.op_builder` | `from cann_ops_transformer.op_builder import OpBuilder, get_as_library` |
| C++ wrapper function name | Same as the schema operator name, placed in `namespace op_api` | `op_api::flash_attn` |
| `PYBIND11_MODULE` export name | Same as the schema operator name | `m.def("flash_attn", &flash_attn, "flash_attn");` |
| Meta implementation function name | Schema operator name plus the `_meta` suffix | `flash_attn_meta` |
| PrivateUse1 dispatcher function name | Underscore prefix plus the schema operator name | `_flash_attn` |
| OpBuilder subclass name | PascalCase of the operator name plus the `OpBuilder` suffix; internal-use classes add a `_` prefix | `FlashAttnOpBuilder`, `_FlashAttnOpBuilder` |
| Public Python interface name | The function name that users call directly; it reflects the usage semantics and carries no `npu_` prefix | `flash_attn` |
| Graph mode GE op function name | PascalCase that matches the GE operator `op_type` | `FlashAttentionScore` |
| Graph mode Converter function name | `convert_` plus the schema operator name | `convert_flash_attn` |

Key naming points:

- **No `npu_` prefix**: public operator names and API interfaces uniformly use the operator semantic name (lowercase snake_case) without backend prefixes such as `npu_`; the schema name, the C++ function name, and the pybind export name must all match this name exactly, otherwise the JIT compilation artifact cannot be called correctly.
- **Interface names reflect semantics**: public function names should stay close to the business semantics. This applies both to operators that simply pass through the aclnn interface (such as `flash_attn`) and to interfaces that wrap extra logic such as struct construction and parameter arrangement; use semantic naming in both cases.
- **Independent aclnn interface names**: the underlying aclnn interfaces follow CANN naming (PascalCase, such as `aclnnFlashAttentionScore`) and stay decoupled from the public operator names; the C++ wrapper calls them through `ACLNN_CMD(aclnnFlashAttentionScore, ...)`.
- **Version suffixes**: different iterative versions of the same operator are distinguished by suffixes such as `_v2` and `_v3`; the schema name, the file name, and the Converter name must carry the version suffix consistently (such as `flash_attn_v2` and `graph_convert_flash_attn_v2.py`).
- **Auxiliary/utility interfaces**: utility functions that accompany the main operator use verb-object semantic naming, such as `get_flash_attn_workspace_size`.

### 2.2 File Naming

- Use **lowercase snake_case** uniformly, with words joined by `_`; uppercase letters, camel case, and hyphens are forbidden.
- The files of each layer for the same operator **share the same main name**, placed under `<category>/<op>/torch_extension/`; only the directory and prefixes distinguish their responsibilities:
  - Python frontend: `<category>/<op>/torch_extension/${op_api}.py`, such as `flash_attn.py`;
  - Operator `__init__.py`: `<category>/<op>/torch_extension/__init__.py`, which defines `__all__` and exports the operator interface;
  - C++ backend: `<category>/<op>/torch_extension/csrc/${op_api}.cpp`, with the same main name as the Python frontend;
  - Graph mode: `<category>/<op>/torch_extension/graph_convert_${op_api}.py`, always with the `graph_convert_` prefix;
  - Documentation: `<category>/<op>/docs/torchapi_${op_api}.md` (in the same directory as the aclnn documentation, with the `torchapi_` prefix in the file name, and the same main name as the public API).
- Place common header files under `torch_extension/cann_ops_transformer/common/`, named by capability domain (such as `aclnn_common.h` and `hccl_common.h`).

### 2.3 Identifier Naming

#### Python Identifiers

- **Functions/variables/parameters**: lowercase snake_case, such as `head_num`, `scale_value`, and `input_layout`.
- **Class names**: PascalCase, such as `OpBuilder` and `FlashAttnOpBuilder`.
- **Module-level constants**: UPPER_SNAKE_CASE, such as `ASCEND_HOME_PATH` and `TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP`.
- **Module-internal private symbols**: mark them with a single underscore `_` prefix, such as `_flash_attn_op_builder` and `_flash_attn`.
- **Type annotations**: public interfaces and key internal functions should carry type annotations (`from typing import Optional, Tuple, List`); uniformly use `Optional[...]` for optional parameters, for example:
  ```python
  def flash_attn(
      query: torch.Tensor,
      key: torch.Tensor,
      value: torch.Tensor,
      atten_mask: Optional[torch.Tensor] = None,
      scale_value: float = 1.0,
      head_num: int = 1,
      input_layout: str = "BSH",
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  ```
- **Parameter naming consistency**: parameters with the same meaning in the schema, meta, dispatcher, public function, and Converter of the same operator should use the same name (such as `head_num`, `scale_value`, and `input_layout`) to avoid inconsistent spellings across layers.

#### C++ Identifiers

- **Functions/local variables/parameters**: lowercase snake_case, such as `head_num`, `scale_value`, and `input_layout_ptr`.
- **Namespaces**: place operator implementations uniformly in `namespace op_api`.
- **Constants**: `const`/`constexpr` constants use UPPER_SNAKE_CASE or PascalCase, such as `const int DIM_THREE = 3;` and `kATenScalarTypeToAclDataTypeTable`.
- **Type aliases/structs**: PascalCase, such as `TensorWrapper` and `TensorListWrapper`.
- **Input parameter type conventions**:
  - Use `const at::Tensor &` for required Tensors;
  - Use `const c10::optional<at::Tensor> &` for optional Tensors;
  - Use `const std::vector<at::Tensor> &` for Tensor lists and `const c10::optional<std::vector<at::Tensor>> &` for optional lists;
  - Use `int64_t` for integer attributes and `c10::optional<int64_t>` for optional integer attributes; use `double` for floating-point attributes; use `std::string` for string attributes.

#### Schema Identifiers (Operator Signature)

- Parameter names use lowercase snake_case and stay consistent with the Python/C++ layers.
- Use `*` to separate positional parameters from keyword parameters: parameters before `*` are required positional parameters, and parameters after `*` are optional keyword parameters (with default values).
- Mark optional parameters with `?` and give their default values, such as `Tensor? atten_mask=None` and `int? head_num=1`; use `Tensor[]` for lists and `Tensor[]?` for optional lists.
- Express multiple outputs as a tuple, such as `-> (Tensor, Tensor, Tensor, Tensor)`.

  The schema of `flash_attn` as an example:

  ```
  flash_attn(Tensor query, Tensor key, Tensor value, *, Tensor? atten_mask=None,
      float scale_value=1.0, int head_num=1, str input_layout="BSH")
      -> (Tensor, Tensor, Tensor, Tensor)
  ```

## 3. Implementation Conventions for Each Layer

<a id="cpp-backend"></a>

### 3.1 C++ Backend (`<category>/<op>/torch_extension/csrc/${op_api}.cpp`)

This layer bridges PyTorch Tensors to the aclnn C-API. Key conventions:

1. The file header includes `#include <torch/extension.h>` and `#include "aclnn_common.h"`, and the implementation is placed in `namespace op_api`.
2. The function signature strictly corresponds to the schema: choose the required/optional parameter types according to [2.3 C++ input parameter type conventions](#c-identifiers).
3. **Input validation**: use `TORCH_CHECK(cond, msg...)` to validate shape, dtype, dimensions, value ranges, and other conditions; error messages must be readable and contain the actual values, for example:
   ```cpp
   TORCH_CHECK((head_num > 0), "The head_num should be greater than 0, current is: ", head_num);
   TORCH_CHECK((query.scalar_type() == key.scalar_type()), "query and key should have the same dtype.");
   ```
4. **Setting the DeviceGuard (critical)**: before you allocate output Tensors, first set `c10::OptionalDeviceGuard` based on the input Tensor to switch the current NPU device to the device of the input Tensor, and use a `{}` scope to wrap the DeviceGuard and the output allocation together (otherwise, when the call runs on a non-default card, output Tensors land on the wrong device, causing device inconsistency):
   ```cpp
   at::Tensor attention_out{nullptr};
   {
       auto local_device = c10::Device(query.device());
       const c10::OptionalDeviceGuard device_guard(local_device);
       attention_out = at::empty(query.sizes(), query.options());
       // ... remaining outputs ...
   }
   ```
5. **Manual output Tensor allocation**: within the scope where the DeviceGuard takes effect, allocate the outputs with `at::empty(...)` using the shape/dtype derived by the meta (standard PyTorch practice), and specify the dtype through `query.options().dtype(...)`.
6. **Launching the kernel**: use the `ACLNN_CMD(aclnn interface name, inputs..., outputs...)` macro to call the aclnn interface (such as `ACLNN_CMD(aclnnFlashAttentionScore, ...)`); the input order must match the aclnn interface definition. This macro automatically completes type conversion, workspace allocation, stream dispatch, and resource release.
7. **Export binding**: bind the C++ function through `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` as a Python interface with the same name as the schema (`m.def("flash_attn", &flash_attn, "flash_attn");`).
8. Express magic numbers (such as the dimension count `3` and the default dtype enum) as named constants; avoid writing bare literals.

   Typical skeleton of the C++ wrapper:

   ```cpp
   std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> flash_attn(
       const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
       const c10::optional<at::Tensor> &atten_mask,
       double scale_value, int64_t head_num, std::string input_layout)
   {
       // 3. Input validation
       TORCH_CHECK((head_num > 0), "The head_num should be greater than 0, current is: ", head_num);

       at::Tensor attention_out{nullptr};
       {
           // 4. DeviceGuard: must come before the output allocation; the scope wraps the output allocation
           auto local_device = c10::Device(query.device());
           const c10::OptionalDeviceGuard device_guard(local_device);
           // 5. Allocate the output Tensors
           attention_out = at::empty(query.sizes(), query.options());
           // ... remaining outputs ...
       }

       // 6. Launch the aclnn kernel
       ACLNN_CMD(aclnnFlashAttentionScore, query, key, value, atten_mask,
           scale_value, head_num, input_layout.data(), /* outputs */ attention_out);

       return std::make_tuple(/* ... */);
   }
   ```

<a id="python-frontend"></a>

### 3.2 Python Frontend (`<category>/<op>/torch_extension/${op_api}.py`)

This layer manages JIT compilation, schema/meta registration, and public interface wrapping:

1. **OpBuilder subclass**: inherit from `OpBuilder`, pass the operator name and category in `__init__` through `super().__init__("<schema operator name>", category="<category>")`, and implement three abstract methods:
    - `sources()`: return the list of C++ source file paths relative to the `cann_ops_transformer` package root, such as `['csrc/attention/flash_attn.cpp']`;
    - `schema()`: return the operator schema string (see [2.3 Schema identifiers](#schema-identifiers-operator-signature));
    - `register_meta()`: register the Meta implementation with `@impl(get_as_library(), self.name, "Meta")`; the Meta only performs shape/dtype derivation and never touches real NPU computation (required for FakeTensor/graph mode). The Meta can also use `torch._check(...)` for constraint validation.
2. **Instantiation and compilation**: instantiate the builder when the module loads and call `_ensure_initialized()` to complete deferred initialization; compilation is triggered by the first `load()`:
    ```python
    _flash_attn_op_builder = _FlashAttnOpBuilder()
    _flash_attn_op_builder._ensure_initialized()
    ```
3. **PrivateUse1 dispatcher**: register the NPU backend implementation with `@impl(get_as_library(), builder.name, "PrivateUse1")`; the function body passes through to the compiled artifact `builder.load().<operator name>(...)`. `PrivateUse1` is the dispatch key that PyTorch reserves for custom NPU backends.
4. **Public interface**: provide the user-facing function `flash_attn(...)`, which handles parameter arrangement, default value processing, and other preparation, and finally calls the dispatcher implementation.
5. **Public APIs must carry comments (docstrings)**: every exported interface must have a docstring that at least covers the function description, the meaning/shape/dtype/value range of each parameter, and the return value description; provide a short usage example when necessary. The docstring content should stay consistent with `<category>/<op>/docs/torchapi_${op_api}.md` to support IDE hints and `help()` inspection. For example:
   ```python
   def flash_attn(
       query: torch.Tensor,
       key: torch.Tensor,
       value: torch.Tensor,
       atten_mask: Optional[torch.Tensor] = None,
       scale_value: float = 1.0,
       head_num: int = 1,
       input_layout: str = "BSH",
   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
       """FlashAttention forward computation, wrapping aclnnFlashAttentionScore.

       Args:
           query (Tensor): Query Tensor. The shape is determined by input_layout (such as BSH), and the dtype supports float16/bfloat16.
           key (Tensor): Key Tensor. The dtype is the same as query.
           value (Tensor): Value Tensor. The dtype is the same as query.
           atten_mask (Tensor, optional): Attention mask. The default None means no mask is used.
           scale_value (float): Scaling factor. The default is 1.0.
           head_num (int): Number of heads per card, that is, the length of the N axis of query. The default is 1.
           input_layout (str): Input data layout. Supports "BSH"/"BNSD" and other layouts. The default is "BSH".

       Returns:
           Tuple[Tensor, Tensor, Tensor, Tensor]:
               softmax_max, softmax_sum, softmax_out, attention_out.
       """
   ```
6. The parameter order and default values of the Meta implementation, the dispatcher, and the public function must all match the schema.

### 3.3 Graph Mode Converter (`<category>/<op>/torch_extension/graph_convert_${op_api}.py`)

This layer converts aten operators to GE nodes in torchair graph mode (GE):

1. **Optional dependency protection**: wrap torchair-related imports uniformly in `try/except ImportError` and control them with the `_TORCHAIR_AVAILABLE` flag to avoid import failures in environments without torchair.
2. **GE op functions**: define a PascalCase function with the same name as `op_type` (such as `FlashAttentionScore`), describe the IR definition of `REG_OP` in the docstring (INPUT/DYNAMIC_INPUT/OPTIONAL_INPUT/OUTPUT/ATTR and other fields), organize `inputs`/`attrs`/`outputs`, and then call `ge_op(...)`; declare the IR through chained `IrDef(...)` calls.
3. **Converter registration**: decorate the `convert_flash_attn` function with `@register_fx_node_ge_converter(torch.ops.cann_ops_transformer.flash_attn.default)`; its parameter order matches the schema exactly, and the function body calls the GE op function above.
4. Export the Converter (such as `convert_flash_attn`) in `ops/__init__.py` to ensure the registration logic runs.

### 3.4 Public Export (Operator `__init__.py` and Package Root `__init__.py`)

Public export has two levels, and the package root no longer requires a manually maintained import list:

1. **`<category>/<op>/torch_extension/__init__.py` (operator level)**: the public interface and Converter of each new operator must be declared in `__all__` and exported through explicit imports. Importing triggers schema/meta/converter registration:
    ```python
    __all__ = ["flash_attn", "convert_flash_attn"]

    from .flash_attn import flash_attn
    from .graph_convert_flash_attn import convert_flash_attn
    ```
    If the operator has additional auxiliary exports (such as `flash_attn_metadata` and `get_symm_buffer_for_mega_moe`), add them to `__all__` and the imports as well.
2. **`ops/__init__.py` (subpackage level)**: this level has been converted to automatic discovery mode (directory scan + entry point), so you do not need to add imports manually. As long as a new operator places files according to the conventions and defines `__all__` in its `__init__.py`, the operator is loaded automatically.
3. **`cann_ops_transformer/__init__.py` (package root level)**: trigger registration through `from . import ops`, and dynamically export the operator interfaces through `__getattr__` and `__dir__`, so that users can access them directly through `cann_ops_transformer.<interface name>`.

## 4. Documentation Conventions (`<category>/<op>/docs/torchapi_${op_api}.md`)

Each public operator API needs a matching Chinese document, placed in the `docs/` directory of the corresponding operator (alongside `aclnn${OpName}.md`, with a file name like `torchapi_${op_api}.md`); align the section order with existing operator documentation (such as `attention/flash_attn/docs/torchapi_flash_attn.md`):

1. **Title**: the operator name (escape special characters such as `_` as `\_`).
2. **Product support**: a table lists the supported product forms (such as `Ascend 950PR&950DT products`) and whether each product is supported.
3. **Function description**: an overview of the API function plus the calculation formulas (use LaTeX for mathematical expressions), and an explanation of the correspondence between each symbol and its parameter.
4. **Function prototype**: a code block gives the complete function signature, including default values and the `*` separator.
5. **Parameter description**: for each parameter, describe whether it is required or optional, its semantics, shape, dtype, data format (such as `$ND$`), whether non-contiguous Tensors are supported, and value ranges/constraints; mark optional parameters with their default values and indicate the items that are not yet supported.
6. **Output description**: for each output, describe the shape, dtype, format, and other attributes.
7. **Constraint description**: list constraints by category, including parameter consistency constraints, shape/value range constraints, quantization scenario constraints, and other constraints; communication operators also need to list communicator constraints.
8. **Supporting interface description**: if the operator must be used together with other interfaces, supplement their prototypes, parameters, and output descriptions.
9. **Invocation example**: provide a complete runnable example for single-operator mode (including multi-card/communicator initialization when necessary); if graph mode is not yet supported, clearly mark the section as "Graph mode invocation (not yet supported)".

## 5. General Coding Constraints

- **License headers**: all new source files (`.py`/`.cpp`/`.h`) must contain the Huawei copyright and the CANN Open Software License Agreement Version 2.0 license header, with the current year. Python/scripts use `#` comments, and C++ uses `//` or `/* */`.
- **Interface comments**: public API interfaces must carry docstrings (function, parameters, return values); see [3.2](#python-frontend). Key C++ wrapper logic (validation, DeviceGuard, aclnn calls) should also carry brief comments.
- **C++ layer DeviceGuard (critical)**: in the C++ wrapper that calls aclnn, use `c10::OptionalDeviceGuard` (constructed from `c10::Device(<input tensor>.device())`) to switch the device to the device of the input Tensor before you allocate output Tensors; see [3.1](#cpp-backend) for details.
- **Front-loaded parameter validation**: use `torch._check(cond, lambda: f"...{var=}...")` on the Python side and `TORCH_CHECK(cond, msg...)` on the C++ side; error messages must contain the actual variable values to ease troubleshooting.
- **Error codes**: on the Python side, use the `ErrCode`/`ops_error` helpers from `torch_npu.utils._error_code` to output standardized error codes, such as `f"... {ops_error(ErrCode.VALUE)}."`.
- **Avoid magic numbers**: express dimension counts, dtype enum values, and similar values as named constants, and explain the enum meanings in documentation/comments (such as `23 → float8_e5m2` and `24 → float8_e4m3fn`).
- **Common capability reuse**: reuse the common headers under `common` first for type conversion, `ACLNN_CMD`, communicator handling, and other capabilities; do not reimplement them in individual operators.
- **Consistency self-check**: before submission, confirm that the operator names, parameter names, parameter order, and default values stay consistent across all seven places: the schema, the C++ wrapper, the Meta, the dispatcher, the public function, the Converter, and the documentation.
