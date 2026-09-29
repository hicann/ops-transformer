# Document Contribution Guide

Welcome to contribute to this project's documentation. High-quality documentation is crucial for project success. This guide helps you efficiently submit documentation that conforms to specifications.

## Contribution Scope

We welcome any contributions that improve documentation quality, including but not limited to:

- Correction and Improvement: Fix typos, grammar errors, incorrect code samples, outdated information, or broken links.

- Clarification and Optimization: Make descriptions clearer and easier to understand, optimize sentence structure, and supplement background knowledge.

- Content Supplement: Add usage samples, API documentation, frequently asked questions (FAQ), best practices, or warning notes for existing features.

- New Content Creation: Write entirely new chapters or tutorials for newly added features, such as operator README, API introduction documents, and so on. If you have questions, it is recommended to create an Issue for discussion first.

- Localization Translation: Help translate or proofread documents in other languages.

- Style and Navigation: Improve the layout, readability, and navigation structure of the documentation website.

## Contribution Process

1. **Preparation**

    - Determine the task: If there is a documentation issue, create a new Issue. The recommended label category is `[Documentation|Document Feedback]`, and provide a detailed description. Based on the existing Issues list, determine the documentation issue to be resolved.
    - Claim the task: Comment `/assign @yourself` under the corresponding Issue to indicate that you will handle it, avoiding duplicate work.

2. **Document Modification**

    - Select a branch: Download the source code from the master or other Tag branch to your local machine.
    - Follow the format:
      - This project recommends using **Markdown format**.
      - Follow the project's existing writing style.
      - Place static resources such as images in the corresponding directory. For example, images are generally placed in the `figures` folder under the docs directory. Special cases can be adjusted as needed.
    - Careful addition and deletion: When modifying content, try to maintain the original line width and line break conventions.

3. **Submit Changes**

    - Atomic commits: Each commit should focus on an independent modification. For example, "Fix spelling errors in the xx guide" and "Update sample code in the API reference" should be two separate commits.

    - Write clear commit messages:

      ```text
      Brief description (no more than 50 characters)

      If necessary, provide a more detailed description here. Explain the reason and content of the modification, rather than what specifically was changed (the code itself shows that).
      Associated Issue: #123
      ```

4. **Submit a Pull Request**

    - Target branch: Merge the PR into the project's target branch.
    - Title and Description:
      - PR title: Should clearly summarize the modification, for example: `[Docs] Fix configuration sample in Quick Start`.
      - PR description: Detail your change content, motivation, and associate the corresponding Issue (use Closes #123 or Fixes #456).
    - Preview check: Check the local or online browsing document effect in advance to ensure rendering meets expectations.
    - Wait for review: Maintainers will review and may provide modification suggestions. Follow up on the discussion promptly.

## Writing Specifications

Before writing project documentation, developers must read the following specifications. If you have questions, you are welcome to propose suggestions at any time!

- Prerequisites: Learn the unified writing specifications provided by the CANN organization first. For details, refer to [CANN Document Writing Specifications](https://gitcode.com/cann/community/blob/master/contributor/docs/document_writing_specs.md).

  - Document content requirements: Introduce the required and optional document deliverables in the project.
  - Directory structure specifications: Introduce the principles of directory division, such as Chinese and English management, and so on.
  - Content element specifications: Introduce the rules for different writing elements, such as file naming, titles, fonts, images, code blocks, links, and so on.

- Precautions:

  In addition to the above writing rules, note the following:

  - Tone: Use a friendly, professional, and neutral tone. For beginners, avoid unnecessary jargon.
  - Terminology: Maintain terminology consistency (such as uniformly using "click" instead of "single click"). Refer to the project terminology table (if available).
  - Code samples:
    - Ensure all code samples are runnable and tested.
    - Provide sufficient context and explanation.
    - Note the environment or prerequisites required for code execution.
  - Punctuation and format:
    - When mixing Chinese and English, use full-width punctuation. Punctuation must conform to the Chinese/English context.
    - Use appropriate heading levels (#, ##, ###).
    - Use lists and tables to organize complex information.
  - Links: Use descriptive link text, avoid "click here", and ensure link resources are real and reliable.
  - Images:
    - Common format: PNG format is recommended. Try to keep the style consistent with existing images.
    - Resolution and clarity: Must be clear and appropriately sized, avoiding blurriness or over-compression.
    - File size: A single image should not exceed 10MB.
  - Copyright: Ensure compliance for all referenced images, literature, and other resources.

## Get Help

If you have any questions during the contribution process:

1. Check existing documentation: If there are issues with templates or specifications, check the project's existing guides, API documentation, or README first.
2. Start a discussion: You can create a new Issue or comment directly in the relevant Issue or PR.

## Operator README Template

For `experimental` newly contributed operators, the operator README is a required document deliverable. You can refer to the **simple template** provided in this section, and you can also expand the content based on this template.

- Document format: Markdown file format is recommended. Native or HTML syntax is supported. Ensure all syntax conforms to official specifications.
- Document purpose: Clearly explain the operator function, implementation principle, parameter specifications, and operator invocation methods.
- Section titles: Prioritize using template section names (such as Function Description, Parameter Description, and so on). The heading level is ##. If there are special cases, increase the level in order. Custom section expansion is supported. Optional sections are presented as needed.
- Content requirements: The writing goal and writing specifications for each chapter are described in detail below. For ease of understanding, the [AddExample](../examples/add_example/README.md) operator README is used as an example.

### Product Support Status

> **Writing Specification**: Recommend table format, list supported product models, and mark √. For product form introduction, refer to [Ascend Product Form Description](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html).

| Product | Supported |
| :----------------------------------------- | :------:|
| Atlas A3 Training Series Products/Atlas A3 Inference Series Products | √ |
| Atlas A2 Training Series Products/Atlas A2 Inference Series Products | √ |

### Function Description

> [!NOTE]
>
> **Writing Goal**: Clarify the operator function, calculation principle, parameter specifications, invocation methods, usage scenarios, and so on.
>
> **Writing Specification**: Recommend unordered list format, generally including the following dimensions
>
> - Operator function (required): Concisely and clearly describe the function in one sentence.
> - Calculation formula (optional): For complex functions, use formulas to introduce the operator implementation principle or calculation process under different scenarios.
> - Other dimensions (optional): Support unordered list expansion. Customize as needed based on actual conditions, such as calculation samples, flowcharts, and so on.

- Operator function: Complete tensor addition calculation.
- Calculation formula:
  $$
  y = x1 + x2
  $$

### Parameter Description

> [!NOTE]
>
> **Writing Goal**: Clarify the meaning, role, specifications, and other information of the parameters defined by the operator.
>
> **Writing Specification**: Use table format, generally including the following dimensions
>
> - Parameter name: Explain the parameters in the operator definition file, keeping the order consistent, such as `op_host/add_example_def.cpp` or `op_graph/add_example_proto.h`.
> - Input/Output/Attribute: Clarify the parameter positioning. Default is required. If optional, it is generally optional input/optional output/optional attribute.
> - Description: Provide the parameter meaning, function, usage scenario introduction, including the mapping relationship with the above formula variables.
> - Data type: The data type supported by the parameter. Tensor data type is generally in `DT_XXX` format. For writing convenience, the `DT_` prefix can be omitted.
> - Data format: The data layout mode supported by the parameter. Tensor format is generally in `FORMAT_xxx` format. For writing convenience, the `FORMAT_` prefix can be omitted.
> - Other dimensions (optional): Support table field expansion. Customize as needed based on actual conditions, such as shape specifications, and so on.

|Parameter Name|Input/Output/Attribute|Description|Data Type|Data Format|
|-----|-----------|----|---------|------|
|x1|Input|Represents the first tensor of the add_example calculation, that is, `x1` in the formula.|FLOAT, FLOAT16, INT32|ND|
|x2|Input|Represents the second tensor of the add_example calculation, that is, `x2` in the formula. The data type is consistent with x1.|ND|
|y|Output|Represents the result tensor of the add_example calculation, that is, `y` in the formula.|FLOAT, FLOAT16, INT32|ND|

### Constraint Description (Optional)

> [!NOTE]
>
> **Writing Goal**: Clarify the precautions during operator usage, such as parameter combination constraints, applicable scenarios, impact on business, operator performance or precision, and so on.
>
> **Writing Specification**: **This section is optional**. If there are no constraints, this section content can be omitted; if there are constraints, use unordered list format.

None

### Invocation Description

> [!NOTE]
>
> **Writing Goal**: Provide operator invocation methods, preferably with sample code that can be directly copied and run for quick verification.
>
> **Writing Specification**: Recommend table format. If the content is complex, other formats can be used.
>
> - Invocation method: Supports aclnn, graph mode, and other invocation methods. You can also customize. Provide at least one method.
> - Sample code: Provide invocation sample code in the operator's `examples` directory, such as `examples/test_aclnn_add_example.cpp`. The file naming rule is test_\$\{invoke\_mode\}\_\${op_name}, where \$\{invoke\_mode\} represents the invocation method and \${op_name} represents the operator name.
> - Description: Supplementary descriptions for different invocation methods, such as invocation scenarios, invocation principles, compilation and execution guidance, and so on. Customize as needed based on actual conditions.

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Invocation Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn Invocation</td>
    <td><a href="../examples/add_example/examples/test_aclnn_add_example.cpp">test_aclnn_add_example</a></td>
    <td rowspan="2">Refer to [Operator Invocation](./en/invocation/quick_op_invocation.md) to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>

### Reference Resources (Optional)

> [!NOTE]
>
> **Writing Goal**: Provide supplementary introductions other than operator function, specifications, and invocation, such as operator design documents (Tiling/Kernel design), reference literature, and so on.
>
> **Writing Specification**: **This section is optional**. If there are no constraints, this section content can be omitted; if there are constraints, use unordered list format.

None
