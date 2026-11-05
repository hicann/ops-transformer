# Contribution Guide

This project welcomes all developers to try it out and contribute. Before participating in community contributions, see [cann-community](https://atomgit.com/cann/community) to learn about the code of conduct, sign the CLA agreement, and familiarize yourself with the contribution process of the source code repository.

## Preparation

When preparing local code and submitting PRs, developers need to pay special attention to the following items:

1. When submitting a PR, carefully fill in the PR template with information such as the business background, purpose, and solution of the PR.
2. If your changes involve new features, new interfaces, new configuration parameters, or modifications to the code flow (that is, changes beyond a simple bug fix), be sure to discuss the solution through an Issue first to avoid having your code rejected. If you are not sure whether a change counts as a "simple bug fix", it is recommended to submit an Issue for discussion.

## Contribution Scenarios

Developer contribution scenarios mainly include:

- [Contributing a New Operator](#1-contributing-a-new-operator)
- [Operator Bug Fixes](#2-operator-bug-fixes)
- [Operator Optimization](#3-operator-optimization)
- [Documentation Corrections](#4-documentation-corrections)
- [Helping Resolve Issues Raised by Others](#5-helping-resolve-issues-raised-by-others)

---

## 1. Contributing a New Operator

The operator development contribution process is as follows:

![Operator development contribution process](./docs/en/figures/op_dev_and_contribute_flow.png "Operator development contribution flowchart")

The ecosystem operators in this repository are custom-developed for specific networks and scenarios and support quick verification and use of customized algorithms. If you have a brand-new operator that you want to design and implement based on the NPU, you are welcome to present your ideas and design proposals in an Issue and contribute the operator to the corresponding operator classification directory under `experimental`. The complete contribution process is as follows:

### 1. Create a Requirement Issue

Create a `Requirement|需求建议` type Issue and describe the design proposal for the new operator. An Issue generally needs to include the following content:

- **Background information**
- **Value/benefits**
- **Design proposal**

For detailed operation steps, refer to: [Issue Operation Guide](https://atomgit.com/cann/community/blob/master/contributor/issue-operation.md).

### 2. Requirement Review

After creating a requirement Issue, apply for a SIG topic and attend the Ops-transformer SIG regular meeting for review at the specified time.

- [Meeting topics and minutes](https://etherpad-cann.meeting.osinfra.cn/p/sig-ops-transformer)
- [Meeting link](https://meeting.osinfra.cn/cann/)

**Handling urgent requirements:**

If a requirement is urgent, you can request an ad-hoc SIG meeting:

1. Fill in the topic application: [Topic application](https://etherpad-cann.meeting.osinfra.cn/p/sig-ops-transformer)
2. Send an email to the [maintainer](https://atomgit.com/cann/community/blob/master/CANN/sigs/ops-transformer/README.md). It is recommended to use an email subject that states the ad-hoc SIG topic application and the topic content

**Requirement acceptance:**

If the requirement is accepted, [SIG members](https://atomgit.com/cann/community/blob/master/CANN/sigs/ops-transformer/README.md) will assign you a suitable operator classification path (for example, `experimental/attention`). Submit the contributed operator to the corresponding operator classification directory under `experimental`.

### 3. PR Submission

#### Deliverables for an Ecosystem Minimal Operator

```text
experimental/${op_class}          # Operator classification (under the experimental directory)
├── ${op_name}                    # Operator name
│   ├── ${op_name}.cpp            # Operator kernel implementation file
│   ├── examples                  # Operator example directory (guards the basic scenario; must be compilable and executable)
│   ├── tests                     # Test directory (containing customized scenario cases)
│   │   └── test_${op_name}.py    # Operator test file
│   ├── CMakeLists.txt            # Operator compilation configuration file
│   └── README.md                 # Operator README document
```

#### Notes on PR Submission

**Preparation before submission:**

- It is recommended to submit the PR only after the requirement review is approved.
- Before submitting a PR, complete the development environment setup and carefully learn the project-specific development specifications and copyright statement requirements (if the code references open-source code snippets, refer to the snippet reference guide). Ensure that your contribution complies with the project standards and sign the CLA. For detailed operation steps, refer to: [PR Operation Guide](https://atomgit.com/cann/community/blob/master/contributor/pull_request_operation.md).

**Deliverable requirements:**

| Type | Requirement | Reference |
|------|------|---------|
| Code deliverables | Provide the operator kernel implementation and the operator test file | [fast_kernel_launch_example](examples/fast_kernel_launch_example/README_en.md) |
| Documentation deliverables | The operator README document is required; other documents are optional depending on the situation | [Documentation Contribution Guide](docs/CONTRIBUTING_DOCS.md) |
| Precision requirement | Newly contributed operators must meet the precision standard | [Open-source precision standard for ecosystem operators](https://atomgit.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| Coverage requirement | The operator test file must cover the core functions and main branches of the operator and meet the UT coverage requirements. Locally, you can run `bash build.sh --experimental -u --cov` to generate an lcov coverage report for self-check. On the CI side, coverage is calculated according to the configuration in [tests/test_config.yaml](tests/test_config.yaml) | [Build and test instructions](docs/zh/install/build.md) |
| Example guarding requirement | Provide an operator example to guard the basic scenario of the operator. The example must be compilable and executable (locally, you can run `bash build.sh --experimental --run_example ${op_name} eager` to compile and run it for verification; for running modes and optional parameters, see `bash build.sh --help`). CI will trigger example verification according to the configuration in [tests/test_config.yaml](tests/test_config.yaml) | [fast_kernel_launch_example](examples/fast_kernel_launch_example/README_en.md) |
| Case submission requirement | For an ecosystem operator customized for specific networks and scenarios, its customized scenario cases (function and precision verification cases under typical network shapes and typical configurations) must be committed to the operator `tests` directory together with the code to ensure sustainable regression verification of the operator. For the case generation rules, refer to the open-source precision standard for ecosystem operators | [Open-source precision standard for ecosystem operators](https://atomgit.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |

**Compliance checks:**

- [ ] Does the code comply with the [C++ Coding Standards](https://atomgit.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)?
- [ ] Does the code compile successfully?
- [ ] Does the Markdown document syntax comply with the specifications?
- [ ] Before committing code with git, you can refer to the [pre-commit tool usage guide](docs/zh/develop/pre-commit_guide.md) to make your code submission more compliant and efficient

**Submission specifications:**

- **Contribution directory**: Submit to the specified directory `experimental/${op_class}` as advised by SIG members. You can refer to the file placement rules of existing operators.
- **PR submission**: Use `git` commands to submit a PR to the target branch. Check whether the PR title is clear, whether the PR description is standard (indicating what is changed and why, and whether the corresponding Issue is linked), and whether the CLA is signed.

> **Description**: If you want to directly contribute a project standard operator, see [PR Submission Requirements for Project Standard Operators](#pr-submission-requirements-for-project-standard-operators) for its deliverables and submission specifications. For the procedure of converting an ecosystem operator into a project standard operator, see [Converting an Ecosystem Operator into a Project Standard Operator](#converting-an-ecosystem-operator-into-a-project-standard-operator).

### 4. CI Gate

Trigger the open-source repository gate by commenting with the `compile` command, and make modifications based on the CI check results. Currently, the CI gate includes the following check items:

- Code compilation
- Static check (if a codecheck false positive is involved, submit it to SIG members for blocking)
- UT tests (including coverage statistics)
- Smoke tests

After the gate is passed, @ the Committer in the associated Issue.

### 5. Committer Review

The Committer will provide review comments after the review. Modify the code according to the comments, and @ the Committer after completion.

### 6. Maintainer Merge

After the Committer review is approved, the `/lgtm` label will be added. The Maintainer performs the final review and, after confirming that there are no issues, adds the `/approve` label to merge the PR.

---

## 2. Operator Bug Fixes

If you find bugs in some operators in this project and want to fix them, you are welcome to create an Issue for feedback and tracking.

You can follow the [Issue submission/handling guide](https://atomgit.com/cann/community#提交Issue处理Issue任务) to create a `Bug-Report|缺陷反馈` type Issue that describes the bug.

---

## 3. Operator Optimization

If you have ideas for generalization enhancements or performance optimizations for some operator implementations in this project and want to implement these optimization points, you are welcome to contribute operator optimizations.

You can follow the [Issue submission/handling guide](https://atomgit.com/cann/community#提交Issue处理Issue任务) to create a `Requirement|需求建议` type Issue that describes the optimization points and provides your design proposal.

---

## 4. Documentation Corrections

If you find description errors in some operator documents in this project, you are welcome to create an Issue for feedback and fixes. For documentation specifications, refer to the [Documentation Contribution Guide](docs/CONTRIBUTING_DOCS.md).

You can follow the [Issue submission/handling guide](https://atomgit.com/cann/community#提交Issue处理Issue任务) to create a `Documentation|文档反馈` type Issue that points out the problems in the corresponding document.

---

## 5. Helping Resolve Issues Raised by Others

If you have a suitable solution to a problem encountered by others in the community, you are welcome to post comments in the Issue to share ideas, help others solve problems and pain points, and jointly improve usability.

If the corresponding Issue requires code changes, you can enter `/assign` or `/assign @yourself` in the Issue comment box to assign the Issue to yourself, and then track and help resolve the problem.

---

## Appendix

### Introduction to Project Standard Operators

Operators in this repository fall into two categories: the ecosystem operators described in the main text, and project standard operators (that is, non-experimental operators). The two categories are compared as follows:

| Comparison dimension | Project standard operator | Ecosystem operator |
|---------|--------------|---------------------|
| Target scenario | Mainstream model scenarios, pursuing high performance, high precision, and high generalization | Custom development for specific networks and scenarios |
| Usage | Released with official versions and supported for use as project standard operators | Supports quick verification and use of customized algorithms |
| Contribution directory | `${op_class}/${op_name}` | `experimental/${op_class}/${op_name}` |
| Deliverable requirements | Complete deliverables (op_host, op_kernel, UT, README, etc.) | Minimal deliverables (kernel implementation, example, test files, and README), plus the coverage, example guarding, and customized scenario case submission requirements |
| Review focus | Comprehensive review of performance, precision, and generalization | Functional correctness, customized scenario adaptation, and completeness of cases and coverage |

### Deliverables for a Project Standard Operator

```text
${op_class}                           # Operator classification
├── ${op_name}                        # Operator name
│   ├── op_host                       # Operator definition and Tiling-related implementation
│   │   ├── ${op_name}_def.cpp        # Operator definition file
│   │   ├── ${op_name}_tiling.cpp     # Operator Tiling implementation file
│   │   ├── ${op_name}_tiling_${sub_case}.cpp  # Optional, sub-scenario Tiling implementation. ${sub_case} indicates the sub-scenario (such as arch35)
│   │   ├── ${op_name}_tiling_${sub_case}.h    # Optional, header file of the sub-scenario Tiling implementation
│   │   └── CMakeLists.txt
│   ├── op_kernel                     # Operator kernel directory
│   │   ├── ${op_name}.cpp            # Kernel entry file, which contains the main function and scheduling logic
│   │   ├── ${op_name}.h              # Kernel implementation file, which defines the kernel header file and contains function descriptions, structure definitions, and logic implementation
│   │   ├── ${op_name}_tiling_data.h  # TilingData file, which stores configuration information related to the Tiling strategy
│   │   └── ${op_name}_tiling_key.h   # TilingKey file, which defines the key of the Tiling strategy to identify different partitioning methods
│   ├── CMakeLists.txt                # Operator compilation configuration file. Keep the original file.
│   ├── README.md                     # Operator description document
│   └── tests                         # Operator test files
│       └── ut                        # Operator UT test files
```

For the complete operator directory structure (including all optional deliverables), see [Project Directory Structure](docs/zh/install/dir_structure.md#项目目录).

> **Description**: For Tiling implementation files in the op_host directory that participate in compilation, the file name must contain the `_tiling` identifier (such as `${op_name}_tiling.cpp` or `${op_name}_tiling_${sub_case}.cpp`). Otherwise, the compilation system will not recognize them. When splitting the Tiling implementation for sub-scenarios (such as a specific architecture like arch35), follow this naming rule.

### PR Submission Requirements for Project Standard Operators

**Deliverable requirements:**

| Type | Requirement | Reference |
|------|------|---------|
| Code deliverables | Provide the operator Tiling implementation in op_host, the operator kernel implementation in op_kernel, and the operator UT test files | [Operator Development Guide](docs/zh/develop/aicore_develop_guide.md) |
| Documentation deliverables | The operator README document is required; other documents are optional depending on the situation | [Documentation Contribution Guide](docs/CONTRIBUTING_DOCS.md) |

**Compliance checks:**

- [ ] Does the code comply with the [C++ Coding Standards](https://atomgit.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)?
- [ ] Does the code comply with the basic programming specifications for standard operators?
- [ ] Does the code compile successfully?
- [ ] Does the Markdown document syntax comply with the specifications?

**Submission specifications:**

- **Contribution directory**: Submit to the specified operator classification directory (such as `attention/${op_name}`) as advised by SIG members. You can refer to the file placement rules of existing operators.
- **PR submission**: Use `git` commands to submit a PR to the target branch. Check whether the PR title is clear, whether the PR description is standard (indicating what is changed and why, and whether the corresponding Issue is linked), and whether the CLA is signed.

### Converting an Ecosystem Operator into a Project Standard Operator

Ecosystem operators are custom-developed for specific networks and scenarios. After an operator becomes stable and mature, you can apply to convert it into a project standard operator. The conversion must meet the following conditions:

- **Project standard operator requirements**: The operator has been stably verified in the target network and scenarios, and its performance and precision meet the usage requirements of project standard operators.
- **Generalization requirement**: The operator has good generalization capability and can support the common usage requirements of mainstream model scenarios.

The conversion procedure is as follows:

1. Create a `Requirement|需求建议` type Issue, describe the actual scenario verification status and generalization performance of the operator, and apply to migrate the operator from `experimental/${op_class}/${op_name}` to the corresponding operator module directory `${op_class}/${op_name}`.
2. Apply for a SIG topic and attend the review: fill in the conversion topic on the [topic application page](https://etherpad-cann.meeting.osinfra.cn/p/sig-ops-transformer) to complete the application, and attend the Ops-transformer SIG regular meeting review as scheduled by the topic ([meeting link](https://meeting.osinfra.cn/cann/)). After the application is accepted by the SIG review, complete the deliverables according to the "Deliverables for a Project Standard Operator" standard (such as the op_host Tiling implementation).
3. The conversion is completed after the CI gate and Committer review are passed and the Maintainer merges the changes. The converted operator will be released with official versions later and supported for use as a project standard operator.
