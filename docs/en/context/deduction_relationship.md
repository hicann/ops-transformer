# deduction relationship

## Deduction Rules

If aclTensor inputs of an API (such as **aclnnAdd** or **aclnnMul**) are of different data types, the API internally deduces a data type and converts the input data into this data type for computation.

Some [data types](./data_type.md) supported by aclTensor meet the following deduction rules, which are similar to those of [Type Promotion](https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc).

> Note:
>
>- For ease of description, the data types used in the table are in abbreviated forms. The meanings of the data types are ACL\_FLOAT\(f32\), ACL\_FLOAT16\(f16\), ACL\_DOUBLE\(f64\), ACL\_BF16\(bf16\), ACL\_INT8\(s8\), ACL\_UINT8\(u8\), ACL\_INT16\(s16\), ACL\_UINT16\(u16\), ACL\_INT32\(s32\), ACL\_UINT32\(u32\), ACL\_INT64\(s64\), ACL\_UINT64\(u64\), ACL\_BOOL\(bool\), ACL\_COMPLEX32\(c32\), ACL\_COMPLEX64\(c64\), and ACL\_COMPLEX128\(c128\).
>- The table heading and the leftmost column in the table indicate the two input data types to be deduced. The corresponding intersections in the table indicate the deduced data types.
>- The cross sign (×) in the table indicates that the two types cannot be deduced.

**Table 1** Data type deduction relationship

| Data Type | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  | bool | c32  | c64  | c128 |
| :------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| **f32**  | f32  | f32  | f64  | f32  | f32  | f32  | f32  |  ×   | f32  |  ×   | f32  |  ×   | f32  | c64  | c64  | c128 |
| **f16**  | f32  | f16  | f64  | f32  | f16  | f16  | f16  |  ×   | f16  |  ×   | f16  |  ×   | f16  | c32  | c64  | c128 |
| **f64**  | f64  | f64  | f64  | f64  | f64  | f64  | f64  |  ×   | f64  |  ×   | f64  |  ×   | f64  | c128 | c128 | c128 |
| **bf16** | f32  | f32  | f64  | bf16 | bf16 | bf16 | bf16 |  ×   | bf16 |  ×   | bf16 |  ×   | bf16 | c32  | c64  | c128 |
|  **s8**  | f32  | f16  | f64  | bf16 |  s8  | s16  | s16  |  ×   | s32  |  ×   | s64  |  ×   |  s8  | c32  | c64  | c128 |
|  **u8**  | f32  | f16  | f64  | bf16 | s16  |  u8  | s16  |  ×   | s32  |  ×   | s64  |  ×   |  u8  | c32  | c64  | c128 |
| **s16**  | f32  | f16  | f64  | bf16 | s16  | s16  | s16  |  ×   | s32  |  ×   | s64  |  ×   | s16  | c32  | c64  | c128 |
| **u16**  |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   | u16  |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |
| **s32**  | f32  | f16  | f64  | bf16 | s32  | s32  | s32  |  ×   | s32  |  ×   | s64  |  ×   | s32  | c32  | c64  | c128 |
| **u32**  |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   | u32  |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |
| **s64**  | f32  | f16  | f64  | bf16 | s64  | s64  | s64  |  ×   | s64  |  ×   | s64  |  ×   | s64  | c32  | c64  | c128 |
| **u64**  |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   | u64  |  ×   |  ×   |  ×   |  ×   |
| **bool** | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  |  ×   | s32  |  ×   | s64  |  ×   | bool | c32  | c64  | c128 |
| **c32**  | c64  | c32  | c128 | c32  | c32  | c32  | c32  |  ×   | c32  |  ×   | c32  |  ×   | c32  | c32  | c64  | c128 |
| **c64**  | c64  | c64  | c128 | c64  | c64  | c64  | c64  |  ×   | c64  |  ×   | c64  |  ×   | c64  | c64  | c64  | c128 |
| **c128** | c128 | c128 | c128 | c128 | c128 | c128 | c128 |  ×   | c128 |  ×   | c128 |  ×   | c128 | c128 | c128 | c128 |

## Deduction Example

- When **aclnnAdd** is called, if the data types of the input parameters are different (one is float16 and the other is float32), the API converts float16 to float32 for computation.
- When **aclnnAdd** is called, if the data types of the input parameters are different (one is float32 and the other is bool), the API converts bool to float32 for computation.
