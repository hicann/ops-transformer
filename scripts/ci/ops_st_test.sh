#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

# Map soc version to arch-specific ST case directory name
declare -A SOC_TO_ST_ARCH
SOC_TO_ST_ARCH=(["ascend910b"]="arch22" ["ascend950"]="arch35")

# Ops that must run kernel cases with dynamic-shape compilation (-d=true, no -b)
# instead of the prebuilt binaries. Keep empty unless an op's installed
# binary_info_config.json cannot be matched by the TTK simplified_key.
KERNEL_DYN_COMPILE_WHITELIST=()

# Check if an op is in the kernel dynamic-compile whitelist
is_op_in_kernel_dyn_whitelist() {
    local op="$1"
    for whitelisted in "${KERNEL_DYN_COMPILE_WHITELIST[@]}"; do
        [[ "${op}" == "${whitelisted}" ]] && return 0
    done
    return 1
}

dotted_line="----------------------------------------------------------------"

# Per-case watchdog in seconds. A hung ttk would otherwise be killed by the CI
# step timeout without ever reaching the failure branch, which loses the run
# log. Set ST_CASE_TIMEOUT=0 to disable.
ST_CASE_TIMEOUT="${ST_CASE_TIMEOUT:-600}"
# Total run budget = planned_cases * ST_CASE_TIMEOUT + reserve, capped by
# ST_MAX_TOTAL_TIMEOUT so the script always terminates before the CI step
# timeout and can still emit the summary / reproduction commands.
ST_MAX_TOTAL_TIMEOUT="${ST_MAX_TOTAL_TIMEOUT:-5400}"
ST_CASE_STARTUP_RESERVE="${ST_CASE_STARTUP_RESERVE:-600}"
ST_TOTAL_DEADLINE=""

# Effective per-case timeout, bounded by the remaining total budget. Returns a
# positive number so the case is always killed; 1 means the budget is gone.
effective_case_timeout() {
    local limit="${ST_CASE_TIMEOUT}"
    if [[ -n "${ST_TOTAL_DEADLINE}" ]]; then
        local remaining=$(( ST_TOTAL_DEADLINE - $(date +%s) ))
        if (( remaining < limit )); then
            limit="${remaining}"
        fi
    fi
    (( limit > 0 )) || limit=1
    echo "${limit}"
}

# Build the timeout prefix for a single case, see effective_case_timeout.
case_timeout_prefix() {
    if [[ ! "${ST_CASE_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
        echo ""
        return 0
    fi
    echo "timeout --signal=TERM --kill-after=30 $(effective_case_timeout) "
}

# Derive the total budget from the number of discovered cases.
init_case_budget() {
    local planned=0
    local arch="${SOC_TO_ST_ARCH[${soc_version}]:-}"
    local op_name code_path op_type ops_test_path cases
    for op_name in "${op_name_array[@]}"; do
        code_path=$(find_op_code_path "${op_name}" 2>/dev/null) || continue
        op_type=$(get_op_type "${code_path}")
        ops_test_path=$(get_ops_test_path "${code_path}" 2>/dev/null) || continue
        cases=$(find_test_cases "${op_name}" "${op_type}" "${arch}")
        planned=$(( planned + $(wc -w <<< "${cases}") ))
        if [[ -n "${test_type_list}" ]]; then
            case ",${test_type_list}," in
                *,e2e,*|*,pta,*)
                    cases=$(find_assets_cases "${ops_test_path}")
                    planned=$(( planned + $(wc -w <<< "${cases}") ));;
            esac
        else
            cases=$(find_assets_cases "${ops_test_path}")
            planned=$(( planned + $(wc -w <<< "${cases}") ))
        fi
    done

    if [[ ! "${ST_CASE_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
        print_msg "case budget: per-case watchdog disabled, no total budget"
        return 0
    fi

    local budget=$(( planned * ST_CASE_TIMEOUT + ST_CASE_STARTUP_RESERVE ))
    if (( budget > ST_MAX_TOTAL_TIMEOUT )); then
        budget="${ST_MAX_TOTAL_TIMEOUT}"
    fi
    ST_TOTAL_DEADLINE=$(( $(date +%s) + budget ))
    print_msg "case budget: ${planned} case(s) x ${ST_CASE_TIMEOUT}s + ${ST_CASE_STARTUP_RESERVE}s reserve -> ${budget}s (cap ${ST_MAX_TOTAL_TIMEOUT}s)"
}

# Python modules imported by the selected operators' TTK test code (assets/st/
# pytest) that are missing in this image. Only a small allowlist is installed
# automatically; unknown modules are reported instead of guessed.
ensure_op_python_deps() {
    local op_dirs=("$@")
    [[ ${#op_dirs[@]} -gt 0 ]] || return 0

    local scan_out
    scan_out=$(python3 - "${framework_path}" "${op_dirs[@]}" <<'PY'
import ast
import importlib.util
import os
import sys

MODULE_TO_PIP = {
    "pandas": "pandas",
    "pytest": "pytest",
    "tensorflow": "tensorflow",
    "ml_dtypes": "ml_dtypes",
    "openpyxl": "openpyxl",
    "psutil": "psutil",
    "yaml": "pyyaml",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "absl": "absl-py",
    "einops": "einops",
    "tqdm": "tqdm",
}
# Modules provided by the ST environment itself (installed wheel / ttk checkout).
IGNORE = {"cann_ops_transformer", "ttk"}


def local_names(root):
    names = set()
    for dirpath, dirnames, filenames in os.walk(root):
        names.update(dirnames)
        names.update(f[:-3] for f in filenames if f.endswith(".py"))
    return names


def module_level_imports(path):
    """Top-level imports only: those are the ones that fail at load time."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            tree = ast.parse(handle.read())
    except Exception:
        return set()
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


root_repo = sys.argv[1]
missing = {}
unknown = set()
for op_dir in sys.argv[2:]:
    tests = os.path.join(op_dir, "tests")
    if not os.path.isdir(tests):
        continue
    # TTK loads the plugin from tests/assets; a few ops additionally reuse
    # helpers from tests/pytest (top level only, subdirs are pytest-only).
    py_files = []
    assets = os.path.join(tests, "assets")
    if os.path.isdir(assets):
        for dirpath, _, filenames in os.walk(assets):
            py_files.extend(os.path.join(dirpath, f) for f in filenames if f.endswith(".py"))
    pytest_dir = os.path.join(tests, "pytest")
    if os.path.isdir(pytest_dir):
        py_files.extend(os.path.join(pytest_dir, f) for f in os.listdir(pytest_dir)
                        if f.endswith(".py"))
    if not py_files:
        continue
    locals_ = set()
    for root in (assets, pytest_dir):
        if os.path.isdir(root):
            locals_ |= local_names(root)
    for py in py_files:
        for mod in module_level_imports(py):
            if mod in locals_ or mod in IGNORE or mod in sys.stdlib_module_names:
                continue
            if has_module(mod):
                continue
            pip_name = MODULE_TO_PIP.get(mod)
            if pip_name:
                missing.setdefault(pip_name, os.path.relpath(py, root_repo))
            else:
                unknown.add(mod)
for pip_name, where in sorted(missing.items()):
    print("%s\t%s" % (pip_name, where))
for mod in sorted(unknown):
    print("UNKNOWN\t%s" % mod)

PY
    )

    local -a packages=()
    local line pkg where
    while IFS=$'\t' read -r pkg where; do
        [[ -z "${pkg}" ]] && continue
        if [[ "${pkg}" == "UNKNOWN" ]]; then
            print_warning "op test imports '${where}' but no pip mapping is known; install it manually if the run fails"
            continue
        fi
        packages+=("${pkg}")
        print_msg "op test dependency missing: ${pkg} (imported by ${where})"
    done <<< "${scan_out}"

    if [[ ${#packages[@]} -eq 0 ]]; then
        print_msg "op test python dependencies: nothing to install"
        return 0
    fi
    print_msg "installing op test dependencies: ${packages[*]}"
    if ! timeout 900 python3 -m pip install --quiet "${packages[@]}"; then
        print_warning "failed to install op test dependencies: ${packages[*]}"
    fi
}

# Print info message with timestamp to stderr
print_msg() {
    local msg="$1"
    local date_time
    date_time=$(date +%Y-%m-%d/%H.%M.%S)
    echo "[INFO]${date_time}: ${msg}" >&2
}

# Print error message with red color to stderr
print_error() {
    echo >&2
    echo $dotted_line >&2
    local msg="$1"
    echo -e "\033[31m[ERROR] ${msg}\033[0m" >&2
    echo $dotted_line >&2
    echo >&2
}

# Print success message with green color to stderr
print_success() {
    echo >&2
    echo $dotted_line >&2
    local msg="$1"
    echo -e "\033[32m[SUCCESS] ${msg}\033[0m" >&2
    echo $dotted_line >&2
    echo >&2
}

# Print warning message with yellow color to stderr
print_warning() {
    echo >&2
    echo $dotted_line >&2
    local msg="$1"
    echo -e "\033[33m[WARNING] ${msg}\033[0m" >&2
    echo $dotted_line >&2
    echo >&2
}

# Dump the tail of a test run log to the job log, so a failed case exposes its
# real error instead of only the pass/fail line.
dump_run_log() {
    local run_log="$1"
    [[ -f "${run_log}" ]] || return 0
    echo $dotted_line >&2
    echo "[DEBUG] tail -n 200 ${run_log}" >&2
    echo $dotted_line >&2
    tail -n 200 "${run_log}" >&2 || true
    echo $dotted_line >&2
    echo >&2
}

# Record a failure that has no usable result csv (watchdog kill, startup
# failure, missing assets plugin, ...). Consumed by print_reproduction_commands
# so those cases still get a reproduction entry.
record_run_failure() {
    local op_name="$1" test_type="$2" testcase="$3" input_csv="$4" cmd="${5:-__NO_CMD__}" reason="$6"
    local record="${log_path}/failed_runs.tsv"
    mkdir -p "${log_path}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${op_name}" "${test_type}" "${testcase}" "${input_csv}" "${cmd}" "${reason}" >> "${record}"
}

# Get op category list from the repo's own module list, fallback to a hardcoded default.
# ops-transformer declares its modules in CMakeLists.txt (ASCEND_ALL_MODULE_NAME);
# ops-nn uses cmake/variables.cmake (OP_CATEGORY_LIST). Support both.
get_op_categories() {
    local categories=""

    local variables_cmake="${framework_path}/cmake/variables.cmake"
    if [[ -f "${variables_cmake}" ]]; then
        categories=$(grep "OP_CATEGORY_LIST" "${variables_cmake}" | \
            sed -n 's/set(OP_CATEGORY_LIST "\(.*\)")/\1/p' | \
            tr -d '"')
    fi

    if [[ -z "${categories}" ]]; then
        local root_cmake="${framework_path}/CMakeLists.txt"
        if [[ -f "${root_cmake}" ]]; then
            categories=$(sed -n \
                's/.*set(ASCEND_ALL_MODULE_NAME[[:space:]]*"\([^"]*\)".*/\1/p' \
                "${root_cmake}" | tr ';' ' ')
        fi
    fi

    if [[ -z "${categories}" ]]; then
        categories="attention moe ffn mhc posembedding gmm mamba mc2"
    fi

    # Keep only real operator categories. Build/test/doc helpers such as
    # examples/ and tests/ must never be treated as an op category, otherwise a
    # bare op name can resolve to e.g. examples/.../grouped_matmul.
    local filtered="" cat
    for cat in ${categories}; do
        case "${cat}" in
            experimental|examples|tests|docs|scripts|cmake|torch_extension|build|build_out|third_party)
                continue
                ;;
        esac
        [[ -d "${framework_path}/${cat}" ]] || continue
        filtered="${filtered:+${filtered} }${cat}"
    done
    if [[ -z "${filtered}" ]]; then
        filtered="attention moe ffn mhc posembedding gmm mamba mc2"
    fi

    categories="${filtered} common"
    echo "${categories}"
}

# Extract op name from a file path based on directory structure
extract_op_from_path() {
    local file_path="$1"
    local op_categories="$2"
    local op_name=""

    local rel_path="${file_path#${framework_path}/}"
    local parts=(${rel_path//\// })

    if [[ ${#parts[@]} -ge 2 ]]; then
        local first_dir="${parts[0]}"
        local second_dir="${parts[1]}"

        local is_category=0
        for cat in ${op_categories}; do
            if [[ "${first_dir}" == "${cat}" ]]; then
                is_category=1
                break
            fi
        done

        if [[ ${is_category} -eq 1 ]]; then
            if [[ "${second_dir}" == "common" ]]; then
                op_name="${first_dir}.common"
            else
                op_name="${second_dir}"
            fi
        elif [[ "${first_dir}" == "experimental" && ${#parts[@]} -ge 3 ]]; then
            local exp_type="${parts[1]}"
            local exp_name="${parts[2]}"
            for cat in ${op_categories}; do
                if [[ "${exp_type}" == "${cat}" ]]; then
                    if [[ "${exp_name}" == "common" ]]; then
                        op_name="experimental/${exp_type}/common"
                    else
                        op_name="experimental/${exp_type}/${exp_name}"
                    fi
                    break
                fi
            done
        fi
    fi

    if [[ -n "${op_name}" ]]; then
        echo "${op_name}"
    fi
}

# Parse changed file list and extract unique op names
parse_ops_from_filelist() {
    local pr_filelist="$1"

    if [[ ! -f "${pr_filelist}" ]]; then
        print_error "pr_filelist not found: ${pr_filelist}"
        return 1
    fi

    local op_categories=$(get_op_categories)
    local changed_files=$(cat "${pr_filelist}" | grep -v '^$' | grep -v '^#' || echo "")

    if [[ -z "${changed_files}" ]]; then
        print_msg "No changed files in pr_filelist"
        return 0
    fi

    local ops_set=""
    while IFS= read -r file_line; do
        [[ -z "${file_line}" ]] && continue
        file_line=$(echo "${file_line}" | sed 's/^[MADRC]\t//')
        # Docs-only changes under experimental/ must not trigger the ST gate:
        # the experimental package has no testable artifact and the experimental
        # path has no st_skip.flag, so ST would fail closed instead of skipping.
        if [[ "${file_line}" == experimental/* && "${file_line}" == *.md ]]; then
            continue
        fi
        local op_name=$(extract_op_from_path "${file_line}" "${op_categories}")
        if [[ -n "${op_name}" ]]; then
            if [[ -z "${ops_set}" ]]; then
                ops_set="${op_name}"
            elif [[ ",${ops_set}," != *",${op_name},"* ]]; then
                ops_set="${ops_set},${op_name}"
            fi
        fi
    done <<< "${changed_files}"

    echo "${ops_set}"
}

# Check mode: parse ops from pr_filelist and decide whether st_950 is needed.
# All logs go to stderr; stdout prints ONLY the machine-readable result line.
run_ci_st_check() {
    local pr_file="$1"

    if [[ ! -f "${pr_file}" || ! -r "${pr_file}" ]]; then
        print_error "ci_st_check: pr_filelist missing or unreadable: ${pr_file}"
        exit 1
    fi

    local ops_list=""
    if ! ops_list=$(parse_ops_from_filelist "${pr_file}"); then
        print_error "ci_st_check: failed to parse pr_filelist: ${pr_file}"
        exit 1
    fi

    if [[ -z "${ops_list}" ]]; then
        print_warning "No operators parsed from ${pr_file}, skip st_950"
        echo "need_st=no"
        return 0
    fi

    # <category>/common/... parses to a pseudo op (<category>.common) that has no
    # op directory and therefore no cases to run. Only keep names that resolve.
    local mainline_ops="" exp_ops="" skipped_ops="" op
    for op in ${ops_list//,/ }; do
        if [[ -z "$(find_op_code_path "${op}" 2>/dev/null)" ]]; then
            skipped_ops="${skipped_ops:+${skipped_ops},}${op}"
            continue
        fi
        if [[ "${op}" == experimental/* ]]; then
            exp_ops="${exp_ops:+${exp_ops},}${op}"
        else
            mainline_ops="${mainline_ops:+${mainline_ops},}${op}"
        fi
    done
    if [[ -n "${skipped_ops}" ]]; then
        print_warning "ci_st_check: no op directory for ${skipped_ops}, skipped"
    fi
    [[ -z "${mainline_ops}" ]] || print_msg "ci_st_check: mainline ops -> ${mainline_ops}"
    [[ -z "${exp_ops}" ]] || print_msg "ci_st_check: experimental ops -> ${exp_ops}"

    # The mainline and experimental packages are built separately and each one
    # only contains its own operators, so report them independently.
    if [[ -n "${mainline_ops}" ]]; then echo "mainline=yes"; else echo "mainline=no"; fi
    if [[ -n "${exp_ops}" ]]; then echo "experimental=yes"; else echo "experimental=no"; fi
    if [[ -n "${mainline_ops}${exp_ops}" ]]; then
        echo "need_st=yes"
    else
        print_warning "ci_st_check: no runnable operator in ${pr_file}, skip st_950"
        echo "need_st=no"
    fi
    return 0
}

# Merge two comma-separated op lists with deduplication
merge_ops_lists() {
    local list1="$1"
    local list2="$2"
    local merged=""

    for op in ${list1//,/ }; do
        if [[ -z "${merged}" ]]; then
            merged="${op}"
        elif [[ ",${merged}," != *",${op},"* ]]; then
            merged="${merged},${op}"
        fi
    done

    for op in ${list2//,/ }; do
        if [[ -z "${merged}" ]]; then
            merged="${op}"
        elif [[ ",${merged}," != *",${op},"* ]]; then
            merged="${merged},${op}"
        fi
    done

    echo "${merged}"
}

# Print usage information and exit
usage() {
    echo "Usage: bash ops_st_test.sh [--soc_version=ascend950] [--ops=op1,op2,op3] [--test_type=kernel,aclnn,e2e] [--pr_filelist=pr_filelist.txt]"
    echo "       bash ops_st_test.sh pr_filelist.txt"
    echo "Options:"
    echo "    --soc_version   (Optional) Specify soc version. Supported: ascend910b, ascend950. If not specified, auto-detect via 'python3 -m ttk info'."
    echo "    --ops           (Optional) Specify operators to test (comma-separated). If not specified, extract from git diff."
    echo "    --test_type     (Optional) Specify test types to run (comma-separated). Supported: kernel, aclnn, e2e. Default: all types."
    echo "    --pr_filelist   (Optional) Path to file containing list of changed files (one per line). If not specified, extract from git diff."
    echo "    --case_path     (Optional) Custom base path for test cases. If specified, st_path will be {case_path}/\${op_type}/\${op_name}"
    echo "    --testcase, -t  (Optional) Specify testcase name(s) to run (comma-separated). Mutually exclusive; cannot specify both or repeat."
    echo "    --update_ttk    (Optional) Force re-download ops-test-kit by removing existing directory and cloning again."
    echo "    --ci_st_check   (Optional) Must be used ONLY with --pr_filelist. Parse pr_filelist and decide whether ST is needed; prints 'need_st=yes|no' to stdout and exits (no env/ttk required)."
    echo "    assets cases    TTK e2e tables under tests/assets[/testcase|/cases] are discovered in addition to tests/st;"
    echo "                    they run with the assets directory as --plugin. Set ST_RUN_PERF_CASES=1 to also run '*perf*' tables."
    echo "    e2e mode        All op categories: tables named '*_graph.csv' run with -c; '*_dyngraph.csv' run with -d."
    echo "                    Graph legs use the ttk default process count (1-4) unless E2E_GRAPH_PC is set;"
    echo "                    other e2e tables keep --pc=16."
    echo "Examples:"
    echo "    bash ops_st_test.sh"
    echo "    bash ops_st_test.sh pr_filelist.txt"
    echo "    bash ops_st_test.sh --soc_version=ascend950"
    echo "    bash ops_st_test.sh --pr_filelist=pr_filelist.txt"
    echo "    bash ops_st_test.sh --soc_version=ascend910b --ops=mat_mul_v3,conv2d_v2"
    echo "    bash ops_st_test.sh --soc_version=ascend950 --test_type=kernel"
    echo "    bash ops_st_test.sh --soc_version=ascend950 --test_type=kernel,e2e"
    echo "    bash ops_st_test.sh --update_ttk"
}

# Detect changed ops via git diff and dependency analysis
get_changed_ops() {
    local base_branch="master"
    local changed_files

    changed_files=$(git diff --name-only "${base_branch}...HEAD" 2>/dev/null || git diff --name-only HEAD~1 HEAD 2>/dev/null || echo "")

    if [[ -z "${changed_files}" ]]; then
        print_msg "No changed files detected"
        return 0
    fi

    mkdir -p "${build_path}"
    cd "${build_path}"
    rm -f CMakeCache.txt
    cmake -DENABLE_EXPERIMENTAL=FALSE -DPREPROCESS_ONLY=ON "${framework_path}" >/dev/null 2>&1 || {
        print_error "Failed to run cmake preprocess"
        return 1
    }
    cd "${framework_path}"

    local changed_files_tmp="${build_path}/tmp/changed_files_tmp.txt"
    mkdir -p "${build_path}/tmp"
    echo "${changed_files}" > "${changed_files_tmp}"

    export BASE_PATH="${framework_path}"
    export BUILD_PATH="${build_path}"

    local is_experimental="FALSE"

    local result
    result=$(python3 "${framework_path}/scripts/util/parse_compile_changed_files.py" "${changed_files_tmp}" "${is_experimental}" 2>&1)
    rm -f "${changed_files_tmp}"

    if [[ -z "${result}" ]]; then
        print_msg "No ops detected from changed files"
        return 0
    fi

    if [[ "${DEBUG_DEPENDENCIES:-}" == "TRUE" ]]; then
        local reverse_deps=$(echo "${result}" | cut -d':' -f1)
        local compile_deps=$(echo "${result}" | cut -d':' -f2)
        print_msg "Reverse dependencies (ops that depend on changed ops): ${reverse_deps}"
        print_msg "Compile dependencies (ops needed to compile): ${compile_deps}"
    fi

    local ops_list=$(echo "${result}" | cut -d':' -f1)
    echo "${ops_list}"
}

# Clone ops-test-kit repository to ttk_path
clone_ops_test_kit() {
    print_msg "Downloading ops-test-kit..."
    git clone https://gitcode.com/cann/ops-test-kit.git "${ttk_path}" || {
        print_error "Failed to clone ops-test-kit"
        exit 1
    }
    print_msg "ops-test-kit downloaded successfully"
}

# Ensure ops-test-kit is available: clone if missing/corrupted, or re-clone if --update_ttk
download_ops_test_kit() {
    print_msg "Preparing build environment..."

    mkdir -p "${build_path}"

    if [[ ! -d "${ttk_path}" ]]; then
        clone_ops_test_kit
    elif [[ ! -f "${ttk_path}/ttk/__init__.py" ]]; then
        print_warning "ops-test-kit directory exists but appears corrupted, re-downloading..."
        rm -rf "${ttk_path}"
        clone_ops_test_kit
    elif [[ "${update_ttk}" == "TRUE" ]]; then
        print_msg "--update_ttk specified, re-downloading ops-test-kit..."
        rm -rf "${ttk_path}"
        clone_ops_test_kit
    else
        print_msg "ops-test-kit already exists, skipping download"
    fi
}

# Detect and validate SOC version via ttk info
detect_soc_version() {
    local ttk_info_output
    ttk_info_output=$(cd "${ttk_path}" && python3 -m ttk info 2>/dev/null || echo "")

    local detected_soc=""
    if echo "${ttk_info_output}" | grep -qi "Ascend910B\|Ascend 910"; then
        detected_soc="ascend910b"
    elif echo "${ttk_info_output}" | grep -qi "Ascend950\|Ascend 950"; then
        detected_soc="ascend950"
    fi

    if [[ -z "${soc_version}" ]]; then
        if [[ -z "${detected_soc}" ]]; then
            print_error "Failed to detect SOC version via 'python3 -m ttk info'. Please specify --soc_version manually."
            exit 1
        fi
        soc_version="${detected_soc}"
        print_msg "Auto-detected soc_version: ${soc_version}"
    else
        if [[ "${soc_version}" != "ascend910b" && "${soc_version}" != "ascend950" ]]; then
            print_error "Unsupported soc_version: ${soc_version}. Supported: ascend910b, ascend950"
            exit 1
        fi

        if [[ -n "${detected_soc}" && "${detected_soc}" != "${soc_version}" ]]; then
            print_error "SOC version mismatch: specified '${soc_version}' but detected '${detected_soc}' from 'python3 -m ttk info'"
            exit 1
        fi
    fi

    print_msg "soc_version: ${soc_version}"
}

# Source CANN environment with fallback chain: ASCEND_HOME_PATH -> ASCEND_TOOLKIT_HOME -> /usr/local/Ascend/cann
setup_cann_env() {
    local cann_dir=""
    local cann_dirs=("${ASCEND_HOME_PATH:-}" "${ASCEND_TOOLKIT_HOME:-}" "/usr/local/Ascend/cann")
    for dir in "${cann_dirs[@]}"; do
        [[ -z "${dir}" ]] && continue
        if [[ -f "${dir}/bin/setenv.bash" ]]; then
            cann_dir="${dir}"
            break
        fi
    done

    if [[ -z "${cann_dir}" ]]; then
        print_error "CANN environment not found. Please set ASCEND_HOME_PATH or install CANN to /usr/local/Ascend/cann"
        exit 1
    fi

    # ASCEND_HOME_PATH 已设置时也要 source，否则 LD_LIBRARY_PATH 缺少 op_tiling 等目录
    print_msg "Sourcing CANN environment from: ${cann_dir}"
    set +u
    source "${cann_dir}/bin/setenv.bash"
    set -u

    # vendor 的 libcust_opmaster 依赖 liboptiling.so，该库只在 op_tiling 目录，显式补齐搜索路径
    local arch
    arch=$(uname -m)
    local tiling_dirs=(
        "${ASCEND_HOME_PATH:-}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/${arch}"
        "${ASCEND_HOME_PATH:-}/opp/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling"
        "${ASCEND_HOME_PATH:-}/opp/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/${arch}"
    )
    local tiling_dir
    for tiling_dir in "${tiling_dirs[@]}"; do
        if [[ -n "${tiling_dir}" && -d "${tiling_dir}" ]]; then
            export LD_LIBRARY_PATH="${tiling_dir}:${LD_LIBRARY_PATH}"
        fi
    done
}

# Find the source directory of an op by name
find_op_code_path() {
    local op_name="$1"
    local code_path=""

    # Path-qualified name (e.g. experimental/attention/quant_flash_attn): resolve as is.
    if [[ "${op_name}" == */* ]]; then
        if [[ -d "${framework_path}/${op_name}" ]]; then
            echo "${framework_path}/${op_name}"
            return 0
        fi
        return 1
    fi

    # Bare name: prefer the mainline op, then experimental/<category>/<op>.
    # Restricting the search to real operator categories keeps helpers such as
    # examples/fast_kernel_launch_example/csrc/grouped_matmul from shadowing
    # the operator of the same name.
    local categories cat candidate
    categories=$(get_op_categories)
    for cat in ${categories}; do
        [[ "${cat}" == "common" ]] && continue
        candidate="${framework_path}/${cat}/${op_name}"
        if [[ -d "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    for candidate in "${framework_path}"/experimental/*/"${op_name}"; do
        if [[ -d "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    # Fallback for layouts that are not <category>/<op> (e.g. nested
    # experimental trees). Non-operator directories stay excluded.
    code_path=$(find "${framework_path}" -type d -name "${op_name}" \
        -not -path "*/build/*" -not -path "*/.git/*" -not -path "*/build_out/*" \
        -not -path "*/third_party/*" -not -path "*/experimental/*" \
        -not -path "*/examples/*" -not -path "*/tests/*" -not -path "*/docs/*" \
        -not -path "*/scripts/*" -not -path "*/cmake/*" -not -path "*/torch_extension/*" \
        2>/dev/null | head -1)
    if [[ -z "${code_path}" ]]; then
        code_path=$(find "${framework_path}" -type d -name "${op_name}" \
            -not -path "*/build/*" -not -path "*/.git/*" -not -path "*/build_out/*" \
            -not -path "*/third_party/*" -not -path "*/examples/*" -not -path "*/tests/*" \
            -not -path "*/docs/*" -not -path "*/scripts/*" -not -path "*/cmake/*" \
            -not -path "*/torch_extension/*" 2>/dev/null | head -1)
    fi

    if [[ -z "${code_path}" ]]; then
        return 1
    fi

    echo "${code_path}"
}

# Get op category type (e.g. matmul, conv) from its source directory path
get_op_type() {
    local code_path="$1"
    local subdir_path=$(realpath "${code_path}")
    local op_type=$(basename "$(dirname "${subdir_path}")")
    echo "${op_type}"
}

# Classify an e2e table into eager / static graph (-c) / dynamic graph (-d).
# All op categories use the same exact suffix convention; non-graph tables stay
# eager. The suffix convention is documented in usage().
classify_e2e_item() {
    local csv_file="$1"

    case "$(basename "${csv_file}" .csv)" in
        *_aclgraph) echo "e2e_aclgraph" ;;
        *_dyngraph) echo "e2e_d" ;;
        *_graph)    echo "e2e_c" ;;
        *)          echo "e2e" ;;
    esac
}

# Exception/invalid case tables assert that bad inputs are rejected; their every
# case is expected to fail. They are not part of the functional ST pass/fail
# gate, so skip them instead of reporting expected failures.
is_invalid_case_table() {
    local csv_base
    csv_base=$(basename "$1" .csv | tr '[:upper:]' '[:lower:]')
    case "${csv_base}" in
        *exception*|*invalid*) return 0 ;;
    esac
    return 1
}

# Find test case CSV files for an op, including arch-specific cases
find_test_cases() {
    local op_name="$1"
    local op_type="$2"
    local arch="$3"
    local test_case_files=()

    local st_path
    if [[ -n "${case_path}" ]]; then
        st_path="${case_path}/${op_type}/${op_name}/"
    else
        st_path="${framework_path}/${op_type}/${op_name}/tests/st"
    fi

    if [[ ! -d "${st_path}" ]]; then
        [[ "${DEBUG_DEPENDENCIES:-}" == "TRUE" ]] && print_msg "No st test directory found for ${op_name} at ${st_path}"
        return 0
    fi

    local all_prefixes=("ttk_kernel" "ttk_aclnn" "ttk_e2e")
    local search_prefixes=()

    if [[ -n "${test_type_list}" ]]; then
        IFS=',' read -r -a input_types <<< "${test_type_list}"
        for input_type in "${input_types[@]}"; do
            # Deprecated alias 'pta' is mapped to 'e2e'
            if [[ "${input_type}" == "pta" ]]; then
                search_prefixes+=("ttk_e2e")
            else
                search_prefixes+=("ttk_${input_type}")
            fi
        done
    else
        search_prefixes=("${all_prefixes[@]}")
    fi

    # Common cases: st/ttk_<type>_*.csv
    for prefix in "${search_prefixes[@]}"; do
        local csv_files=$(find "${st_path}" -maxdepth 1 -name "${prefix}_*.csv" -type f 2>/dev/null)
        for csv_file in ${csv_files}; do
            local test_type="${prefix#ttk_}"
            if [[ "${test_type}" == "e2e" ]]; then
                test_type=$(classify_e2e_item "${csv_file}")
            fi
            test_case_files+=("${test_type}:${csv_file}")
        done
    done

    # Arch-specific cases: st/<arch>/ttk_<type>_*.csv
    if [[ -n "${arch}" ]]; then
        local arch_path="${st_path}/${arch}"
        if [[ -d "${arch_path}" ]]; then
            for prefix in "${search_prefixes[@]}"; do
                local csv_files=$(find "${arch_path}" -maxdepth 1 -name "${prefix}_*.csv" -type f 2>/dev/null)
                for csv_file in ${csv_files}; do
                    local test_type="${prefix#ttk_}"
                    if [[ "${test_type}" == "e2e" ]]; then
                        test_type=$(classify_e2e_item "${csv_file}")
                    fi
                    test_case_files+=("${test_type}:${csv_file}")
                done
            done
        fi
    fi

    local filtered_case_files=()
    local case_item
    if [[ ${#test_case_files[@]} -gt 0 ]]; then
        for case_item in "${test_case_files[@]}"; do
            if is_invalid_case_table "${case_item#*:}"; then
                print_msg "Skipping invalid/exception case table: ${case_item#*:}"
                continue
            fi
            filtered_case_files+=("${case_item}")
        done
    fi

    echo "${filtered_case_files[*]}"
}

# Case tables kept under tests/assets (assets root, assets/testcase or
# assets/cases) instead of tests/st. They use the api_name-based e2e schema and
# the assets directory itself is the TTK plugin. Returned in the same
# 'test_type:csv' form as find_test_cases, with type 'e2e_assets'.
ASSETS_CASE_SUBDIRS=("" "testcase" "cases")

find_assets_cases() {
    local ops_test_path="$1"
    local case_files=()

    local assets_path="${ops_test_path}/assets"
    if [[ ! -d "${assets_path}" ]]; then
        return 0
    fi

    for sub in "${ASSETS_CASE_SUBDIRS[@]}"; do
        local case_dir="${assets_path}"
        if [[ -n "${sub}" ]]; then
            case_dir="${assets_path}/${sub}"
        fi
        [[ -d "${case_dir}" ]] || continue

        local csv_files
        csv_files=$(find "${case_dir}" -maxdepth 1 -name "*.csv" -type f 2>/dev/null | sort)
        for csv_file in ${csv_files}; do
            local csv_base
            csv_base=$(basename "${csv_file}")
            # ttk_*.csv is the st/ convention; ignore it here so the same table
            # is not executed twice when an op carries both layouts.
            [[ "${csv_base}" == ttk_* ]] && continue
            # Perf-only case tables are not part of the functional ST gate.
            if [[ "${ST_RUN_PERF_CASES:-0}" != "1" && "${csv_base}" == *perf* ]]; then
                print_msg "Skipping perf case table: ${csv_file}"
                continue
            fi
            if is_invalid_case_table "${csv_file}"; then
                print_msg "Skipping invalid/exception case table: ${csv_file}"
                continue
            fi
            case_files+=("e2e_assets:${csv_file}")
        done
    done

    echo "${case_files[*]}"
}

# Get the tests directory path for an op
get_ops_test_path() {
    local code_path="$1"
    local ops_test_path="${code_path}/tests"

    if [[ ! -d "${ops_test_path}" ]]; then
        print_msg "No tests directory found under ${code_path}"
        return 1
    fi

    echo "${ops_test_path}"
}

# Check precision status of test results in a CSV file
check_precision_status() {
    local result_csv="$1"
    local op_name="$2"
    local testcase_name="$3"

    if [[ ! -f "${result_csv}" ]]; then
        print_warning "Result csv file not found: ${result_csv}"
        return 1
    fi

    python3 "${framework_path}/scripts/ci/ops_test_util.py" \
        --action=check_precision \
        --result_csv="${result_csv}" \
        --op_name="${op_name}" \
        --testcase_name="${testcase_name}"

    return $?
}

# Verify that plugin assets directory contains required .py files
check_plugin_assets() {
    local plugin_path="$1"
    local op_name="$2"

    local assets_path="${plugin_path}/assets"

    if [[ ! -d "${assets_path}" ]]; then
        print_warning "assets directory not found for ${op_name}: ${assets_path}"
        return 1
    fi

    local py_files=$(find "${assets_path}" -maxdepth 1 -name "*.py" -type f 2>/dev/null | head -1)
    if [[ -z "${py_files}" ]]; then
        print_warning "No .py files found in assets directory for ${op_name}: ${assets_path}"
        return 1
    fi

    return 0
}

# Run kernel-level test for an op using ttk kernel mode
run_kernel_test() {
    local op_name="$1"
    local test_csv="$2"
    local ops_test_path="$3"

    if [[ ! -f "${test_csv}" ]]; then
        print_warning "Test csv file not found: ${test_csv}, skipping this test case"
        return 0
    fi

    if [[ ! -d "${ops_test_path}" ]]; then
        print_warning "Plugin directory not found: ${ops_test_path}, skipping this test case"
        return 0
    fi

    if ! check_plugin_assets "${ops_test_path}" "${op_name}"; then
        return 0
    fi

    local testcase_name=$(basename "${test_csv}" .csv)
    local log_op_dir="${log_path}/${op_name}"
    mkdir -p "${log_op_dir}"

    print_msg "Running kernel test for ${op_name}, testcase: ${testcase_name}"

    cd "${ttk_path}"

    local testcase_arg=""
    if [[ -n "${testcase_filter}" ]]; then
        testcase_arg="-t ${testcase_filter}"
    fi

    # Whitelisted ops run with dynamic-shape compilation to bypass the
    # simplified_key format normalization mismatch on installed binary configs.
    local compile_flags="-d=false -b=release"
    if is_op_in_kernel_dyn_whitelist "${op_name}"; then
        compile_flags="-d=true"
    fi
    local cmd="$(case_timeout_prefix)python3 -m ttk kernel -i ${test_csv} -o ${log_op_dir}/${testcase_name}_result.csv --plugin ${ops_test_path} ${compile_flags} --pc=16 --run 1 --compare close --task-prof false --no-memory-check ${testcase_arg}"
    print_msg "Executing: ${cmd}"

    local start_time=$(date +%s)
    ${cmd} 2>&1 | tee "${log_op_dir}/${testcase_name}_run.log" >&2
    local test_failed=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    if [[ ${test_failed} -ne 0 ]]; then
        print_error "kernel test failed for ${op_name}, testcase: ${testcase_name}, elapsed: ${elapsed}s"
        record_run_failure "${op_name}" "kernel" "${testcase_name}" "${test_csv}" "${cmd}" "exit_code_${test_failed}"
        dump_run_log "${log_op_dir}/${testcase_name}_run.log"
    else
        print_msg "kernel test completed for ${op_name}, testcase: ${testcase_name}, elapsed: ${elapsed}s"
    fi

    local result_csv="${log_op_dir}/${testcase_name}_result.csv"
    echo "${result_csv}"

    if [[ ${test_failed} -ne 0 ]]; then
        return 1
    fi
}

# Run aclnn API test for an op using ttk aclnn mode
run_aclnn_test() {
    local op_name="$1"
    local test_csv="$2"
    local ops_test_path="$3"

    if [[ ! -f "${test_csv}" ]]; then
        print_warning "Test csv file not found: ${test_csv}, skipping this test case"
        return 0
    fi

    if [[ ! -d "${ops_test_path}" ]]; then
        print_warning "Plugin directory not found: ${ops_test_path}, skipping this test case"
        return 0
    fi

    if ! check_plugin_assets "${ops_test_path}" "${op_name}"; then
        return 0
    fi

    local testcase_name=$(basename "${test_csv}" .csv)
    local log_op_dir="${log_path}/${op_name}"
    mkdir -p "${log_op_dir}"

    print_msg "Running aclnn test for ${op_name}, testcase: ${testcase_name}"

    cd "${ttk_path}"

    local testcase_arg=""
    if [[ -n "${testcase_filter}" ]]; then
        testcase_arg="-t ${testcase_filter}"
    fi
    local cmd="$(case_timeout_prefix)python3 -m ttk aclnn -i ${test_csv} -o ${log_op_dir}/${testcase_name}_result.csv --plugin ${ops_test_path} --pc=16 --run 1 --compare close --task-prof false --no-memory-check ${testcase_arg}"
    print_msg "Executing: ${cmd}"

    local start_time=$(date +%s)
    ${cmd} 2>&1 | tee "${log_op_dir}/${testcase_name}_run.log" >&2
    local test_failed=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    if [[ ${test_failed} -ne 0 ]]; then
        print_error "aclnn test failed for ${op_name}, testcase: ${testcase_name}, elapsed: ${elapsed}s"
        record_run_failure "${op_name}" "aclnn" "${testcase_name}" "${test_csv}" "${cmd}" "exit_code_${test_failed}"
        dump_run_log "${log_op_dir}/${testcase_name}_run.log"
    else
        print_msg "aclnn test completed for ${op_name}, testcase: ${testcase_name}, elapsed: ${elapsed}s"
    fi

    local result_csv="${log_op_dir}/${testcase_name}_result.csv"
    echo "${result_csv}"

    if [[ ${test_failed} -ne 0 ]]; then
        return 1
    fi
}

# Run end-to-end test for an op using ttk e2e mode
run_e2e_test() {
    local op_name="$1"
    local test_csv="$2"
    local ops_test_path="$3"
    local e2e_mode="${4:-}"

    if [[ ! -f "${test_csv}" ]]; then
        print_warning "Test csv file not found: ${test_csv}, skipping this test case"
        return 0
    fi

    if [[ ! -d "${ops_test_path}" ]]; then
        print_warning "Plugin directory not found: ${ops_test_path}, skipping this test case"
        return 0
    fi

    if ! check_plugin_assets "${ops_test_path}" "${op_name}"; then
        return 0
    fi

    local testcase_name=$(basename "${test_csv}" .csv)
    local log_op_dir="${log_path}/${op_name}"
    mkdir -p "${log_op_dir}"

    print_msg "Running e2e test for ${op_name}, testcase: ${testcase_name}"

    cd "${ttk_path}"

    local testcase_arg=""
    if [[ -n "${testcase_filter}" ]]; then
        testcase_arg="-t ${testcase_filter}"
    fi
    # Graph legs (-c/-d) use the ttk default process count (1-4, capped at 4)
    # unless E2E_GRAPH_PC is set: multiple workers only parallelize host-side
    # input/golden generation (device execution is serialized by DeviceLock),
    # and a high --pc was observed to trigger 507018 ACL stream pseudo-failures.
    local mode_flag=""
    local pc_arg="--pc=16"
    if [[ -n "${e2e_mode}" ]]; then
        if [[ "${e2e_mode}" == "aclgraph" ]]; then
            mode_flag="--aclgraph"
        else
            mode_flag="-${e2e_mode}"
        fi
        if [[ -n "${E2E_GRAPH_PC:-}" ]]; then
            pc_arg="--pc=${E2E_GRAPH_PC}"
        else
            pc_arg=""
        fi
    fi
    local cmd="$(case_timeout_prefix)python3 -m ttk e2e -i ${test_csv} -o ${log_op_dir}/${testcase_name}_result.csv --plugin ${ops_test_path} ${mode_flag} ${pc_arg} --run 1 --compare close --task-prof false --no-memory-check ${testcase_arg}"
    print_msg "Executing: ${cmd}"

    local start_time=$(date +%s)
    ${cmd} 2>&1 | tee "${log_op_dir}/${testcase_name}_run.log" >&2
    local test_failed=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    if [[ ${test_failed} -ne 0 ]]; then
        print_error "e2e test failed for ${op_name}, testcase: ${testcase_name}, elapsed: ${elapsed}s"
        record_run_failure "${op_name}" "e2e" "${testcase_name}" "${test_csv}" "${cmd}" "exit_code_${test_failed}"
        dump_run_log "${log_op_dir}/${testcase_name}_run.log"
    else
        print_msg "e2e test completed for ${op_name}, testcase: ${testcase_name}, elapsed: ${elapsed}s"
    fi

    local result_csv="${log_op_dir}/${testcase_name}_result.csv"
    echo "${result_csv}"

    if [[ ${test_failed} -ne 0 ]]; then
        return 1
    fi
}

# Run an assets-based case table (api_name schema) with the assets directory as
# the TTK plugin. These specs ship their own compare/pre_compare callbacks, so no
# --compare flag is passed: the tolerance comes from the CSV or the spec.
run_assets_test() {
    local op_name="$1"
    local test_csv="$2"
    local ops_test_path="$3"

    if [[ ! -f "${test_csv}" ]]; then
        print_warning "Test csv file not found: ${test_csv}, skipping this test case"
        return 0
    fi

    local testcase_name=$(basename "${test_csv}" .csv)
    local plugin_dir="${ops_test_path}/assets"
    if [[ -z "$(find "${plugin_dir}" -maxdepth 1 -name "*.py" -type f 2>/dev/null | head -1)" ]]; then
        print_error "No TTK plugin .py found in ${plugin_dir}; assets table ${testcase_name} cannot run"
        record_run_failure "${op_name}" "e2e" "${testcase_name}" "${test_csv}" "" "missing_assets_plugin"
        return 1
    fi
    local log_op_dir="${log_path}/${op_name}"
    mkdir -p "${log_op_dir}"

    print_msg "Running assets e2e test for ${op_name}, testcase: ${testcase_name}"

    cd "${ttk_path}"

    local testcase_arg=""
    if [[ -n "${testcase_filter}" ]]; then
        testcase_arg="-t ${testcase_filter}"
    fi
    local cmd="env PYTHONPATH=${plugin_dir}${PYTHONPATH:+:${PYTHONPATH}} $(case_timeout_prefix)python3 -m ttk e2e -i ${test_csv} -o ${log_op_dir}/${testcase_name}_result.csv --plugin ${plugin_dir} --pc=16 --run 1 --task-prof false --no-memory-check ${testcase_arg}"
    print_msg "Executing: ${cmd}"

    local start_time=$(date +%s)
    ${cmd} 2>&1 | tee "${log_op_dir}/${testcase_name}_run.log" >&2
    local test_failed=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    if [[ ${test_failed} -ne 0 ]]; then
        print_error "assets e2e test failed for ${op_name}, testcase: ${testcase_name}, elapsed: ${elapsed}s"
        record_run_failure "${op_name}" "e2e" "${testcase_name}" "${test_csv}" "${cmd}" "exit_code_${test_failed}"
        dump_run_log "${log_op_dir}/${testcase_name}_run.log"
    else
        print_msg "assets e2e test completed for ${op_name}, testcase: ${testcase_name}, elapsed: ${elapsed}s"
    fi

    local result_csv="${log_op_dir}/${testcase_name}_result.csv"
    echo "${result_csv}"

    if [[ ${test_failed} -ne 0 ]]; then
        return 1
    fi
}

# Summarize test results to CSV and print pass rate for an op
summarize_op_results() {
    local op_name="$1"
    local test_type="$2"
    local result_csvs="$3"

    local summary_file="${log_path}/${test_type}_summary.csv"

    if [[ -z "${result_csvs}" ]]; then
        return 0
    fi

    for result_csv in ${result_csvs}; do
        if [[ ! -f "${result_csv}" ]]; then
            continue
        fi

        python3 "${framework_path}/scripts/ci/ops_test_util.py" \
            --action=summarize \
            --result_csv="${result_csv}" \
            --op_name="${op_name}" \
            --test_type="${test_type}" \
            --summary_file="${summary_file}"
    done

    local total_cases=0
    local passed_cases=0
    # Summary rows carry extra precision columns (dyn_prec/cst_prec/bin_prec)
    # appended by ops_test_util.py; keep them out of the status field.
    if [[ -f "${summary_file}" ]]; then
        while IFS=',' read -r csv_op csv_testcase csv_type csv_result csv_status _rest; do
            if [[ "${csv_op}" == "${op_name}" && "${csv_type}" == "${test_type}" ]]; then
                ((++total_cases))
                if [[ "${csv_status}" == "PASS" ]]; then
                    ((++passed_cases))
                fi
            fi
        done < <(tail -n +2 "${summary_file}")
    fi

    if [[ ${total_cases} -gt 0 ]]; then
        print_msg "${op_name} ${test_type}: ${passed_cases}/${total_cases} passed"
    fi
}

# Print a formatted summary table of all test results
print_summary_table() {
    python3 "${framework_path}/scripts/ci/ops_test_util.py" \
        --action=print_table \
        --log_path="${log_path}"
}

# Generate and print reproduction commands for failed test cases
print_reproduction_commands() {
    local repro_file="${log_path}/repro_commands.txt"
    : > "${repro_file}"
    local has_failed=0

    {
        echo ""
        echo "${dotted_line}"
        echo "[ERROR] Reproduction commands for failed test cases:"
        echo "${dotted_line}"
        echo ""
    } >&2

    # Failures recorded at execution time (watchdog kill / startup failure /
    # missing plugin) have no result csv, so they never reach the summary.
    local seen_keys=" " failed_runs="${log_path}/failed_runs.tsv"
    local rec_op rec_type rec_case rec_csv rec_cmd rec_reason
    if [[ -f "${failed_runs}" ]]; then
        while IFS=$'\t' read -r rec_op rec_type rec_case rec_csv rec_cmd rec_reason; do
            [[ -z "${rec_op}" ]] && continue
            # Bash merges consecutive tab IFS delimiters; record_run_failure
            # stores __NO_CMD__ for an empty command so the reason column stays put.
            [[ "${rec_cmd}" == "__NO_CMD__" ]] && rec_cmd=""
            has_failed=1
            seen_keys="${seen_keys}${rec_op}|${rec_case} "
            {
                echo "  # [${rec_type}] ${rec_op} / ${rec_case} (${rec_reason})"
                if [[ -n "${rec_cmd}" ]]; then
                    echo "  ${rec_cmd}"
                else
                    echo "  # no command recorded, check case table: ${rec_csv}"
                fi
                echo ""
            } >&2
            {
                echo "[${rec_type}] ${rec_op} / ${rec_case} (${rec_reason})"
                [[ -n "${rec_cmd}" ]] && echo "${rec_cmd}"
                echo "table: ${rec_csv}"
                echo ""
            } >> "${repro_file}"
        done < "${failed_runs}"
    fi

    local summary_files=("kernel_summary.csv" "aclnn_summary.csv" "e2e_summary.csv")
    for sf in "${summary_files[@]}"; do
        local summary_path="${log_path}/${sf}"
        [[ ! -f "${summary_path}" ]] && continue
        local test_type="${sf%%_summary.csv}"

        while IFS=',' read -r f_op f_tc f_tt f_csv f_status _rest || [[ -n "${f_op}" ]]; do
            [[ "${f_op}" == "op_name" ]] && continue
            [[ "${f_status}" == "PASS" ]] && continue
            [[ -z "${f_op}" ]] && continue
            # already emitted from failed_runs.tsv
            [[ "${seen_keys}" == *"${f_op}|${f_tc} "* ]] && continue

            has_failed=1

            local code_path="" op_type="" ops_test_path="" input_csv=""
            code_path=$(find_op_code_path "${f_op}" 2>/dev/null) || code_path=""
            if [[ -n "${code_path}" ]]; then
                op_type=$(get_op_type "${code_path}")
                ops_test_path=$(get_ops_test_path "${code_path}" 2>/dev/null) || ops_test_path=""
            fi

            local result_basename csv_basename
            result_basename=$(basename "${f_csv}")
            csv_basename="${result_basename%_result.csv}"
            if [[ -n "${csv_basename}" ]]; then
                # Case tables may live under --case_path (e.g. opstest); fall back
                # to the op repo when the table is not there.
                local search_root="${case_path:-${framework_path}}"
                if [[ -d "${search_root}" ]]; then
                    input_csv=$(find "${search_root}" -name "${csv_basename}.csv" -type f 2>/dev/null | head -1 || true)
                fi
                if [[ -z "${input_csv}" && -n "${case_path}" && -d "${framework_path}" ]]; then
                    input_csv=$(find "${framework_path}" -name "${csv_basename}.csv" -type f 2>/dev/null | head -1 || true)
                fi
            fi

            local ttk_mode extra_flags pc_flag="--pc=16"
            case "${test_type}" in
                kernel) ttk_mode="kernel"
                        if is_op_in_kernel_dyn_whitelist "${f_op}"; then
                            extra_flags="-d=true"
                        else
                            extra_flags="-d=false -b=release"
                        fi;;
                aclnn)  ttk_mode="aclnn";  extra_flags="";;
                e2e)    ttk_mode="e2e";    extra_flags=""
                        # assets tables run through their own spec, never as graph legs
                        if [[ -n "${csv_basename}" && "${input_csv}" != */assets/* ]]; then
                            local e2e_kind
                            e2e_kind=$(classify_e2e_item "${csv_basename}.csv")
                            case "${e2e_kind}" in
                                e2e_c) extra_flags="-c" ;;
                                e2e_d) extra_flags="-d" ;;
                                e2e_aclgraph) extra_flags="--aclgraph" ;;
                            esac
                        fi
                        if [[ -n "${extra_flags}" ]]; then
                            if [[ -n "${E2E_GRAPH_PC:-}" ]]; then
                                pc_flag="--pc=${E2E_GRAPH_PC}"
                            else
                                pc_flag=""
                            fi
                        fi;;
                *)      ttk_mode="${test_type}"; extra_flags="";;
            esac

            # assets cases use the assets dir as plugin and rely on the spec's own
            # compare callback, so no --compare is passed (see run_assets_test).
            local plugin_path="${ops_test_path}" compare_flag="--compare close"
            if [[ "${input_csv}" == */assets/* ]]; then
                plugin_path="${ops_test_path}/assets"
                compare_flag=""
            fi

            local repro_cmd="cd ${ttk_path} && env PYTHONPATH=${plugin_path}:\${PYTHONPATH:-} python3 -m ttk ${ttk_mode} -i ${input_csv} -o ${f_csv} --plugin ${plugin_path} ${extra_flags} ${pc_flag} --run 1 ${compare_flag} --task-prof false --no-memory-check -t ${f_tc}"

            {
                echo "  # [${test_type}] ${f_op} / ${f_tc}"
                echo "  ${repro_cmd}"
                echo ""
            } >&2

            echo "[${test_type}] ${f_op} / ${f_tc}" >> "${repro_file}"
            echo "${repro_cmd}" >> "${repro_file}"
            echo "" >> "${repro_file}"
        done < "${summary_path}"
    done

    if [[ ${has_failed} -eq 0 ]]; then
        echo "  (no failed test cases found)" >&2
    fi
    echo "${dotted_line}" >&2
    print_msg "Reproduction commands also saved to: ${repro_file}"
}

# Some experimental ops ship their torch adapter as <op>/torch_ops_extension
# instead of the standard <op>/torch_extension. The repo-level
# `build.sh --torch_extension_only` only collects the latter, so these ops are
# missing from the cann_ops_transformer wheel and their assets cases fail with
# "No module named 'custom_ops'". Build and install the per-op extension for
# the selected experimental op here; mainline ops are left untouched.
build_experimental_torch_ops_extension() {
    local op_name="$1"
    local code_path="$2"

    case "${code_path}" in
        "${framework_path}/experimental/"*) ;;
        *) return 0 ;;
    esac

    local ext_dir="${code_path}/torch_ops_extension"
    [[ -f "${ext_dir}/setup.py" ]] || return 0

    local log_file="${log_path}/$(echo "${op_name}" | tr '/' '_')_torch_ops_extension.log"
    print_msg "Building per-op torch extension: ${ext_dir}"
    if ( cd "${ext_dir}" && bash build_and_install.sh ) > "${log_file}" 2>&1; then
        print_success "per-op torch extension installed for ${op_name}"
        return 0
    fi

    print_error "Failed to build per-op torch extension for ${op_name}, see ${log_file}"
    tail -n 40 "${log_file}" >&2 || true
    return 1
}

# Run all test types for a single op and collect results
run_single_op_test() {
    local op_name="$1"

    print_msg "=== Testing op: ${op_name} ==="

    local code_path=$(find_op_code_path "${op_name}")
    if [[ -z "${code_path}" ]]; then
        print_warning "Cannot find op directory for ${op_name}, skipping"
        return 0
    fi

    local op_type=$(get_op_type "${code_path}")
    print_msg "op_type: ${op_type}, op_name: ${op_name}"

    local arch="${SOC_TO_ST_ARCH[${soc_version}]:-}"
    if [[ -n "${arch}" ]]; then
        print_msg "soc_version: ${soc_version}, arch: ${arch}"
    fi

    local ops_test_path=$(get_ops_test_path "${code_path}")
    if [[ -z "${ops_test_path}" ]]; then
        print_msg "No tests directory found, skipping ${op_name}"
        return 0
    fi

    local test_cases=$(find_test_cases "${op_name}" "${op_type}" "${arch}")

    # assets-based tables are e2e only; honour --test_type when it excludes e2e
    local want_e2e=1
    if [[ -n "${test_type_list}" ]]; then
        want_e2e=0
        case ",${test_type_list}," in
            *,e2e,*|*,pta,*) want_e2e=1 ;;
        esac
    fi

    local assets_cases=""
    if [[ ${want_e2e} -eq 1 ]]; then
        assets_cases=$(find_assets_cases "${ops_test_path}")
    fi
    if [[ -n "${assets_cases}" ]]; then
        test_cases="${test_cases} ${assets_cases}"
    fi

    if [[ -z "${test_cases}" ]]; then
        print_msg "No test cases found for ${op_name}"
        return 0
    fi

    if ! build_experimental_torch_ops_extension "${op_name}" "${code_path}"; then
        return 1
    fi

    local result_csvs=()
    local kernel_csvs=()
    local aclnn_csvs=()
    local e2e_csvs=()
    local test_case_array=(${test_cases})
    local result_csv
    local testcase_name
    local op_error_flag=0

    declare -A test_runners=(
        ["kernel"]="run_kernel_test"
        ["aclnn"]="run_aclnn_test"
        ["e2e"]="run_e2e_test"
        ["e2e_c"]="run_e2e_test"
        ["e2e_d"]="run_e2e_test"
        ["e2e_aclgraph"]="run_e2e_test"
        ["e2e_assets"]="run_assets_test"
    )

    for test_item in "${test_case_array[@]}"; do
        local test_type=$(echo "${test_item}" | cut -d':' -f1)
        local test_csv=$(echo "${test_item}" | cut -d':' -f2-)
        local test_ret=0

        if [[ -z "${test_runners[${test_type}]+_}" ]]; then
            print_warning "Unknown test type: ${test_type}, skipping"
            continue
        fi

        local runner_func="${test_runners[${test_type}]}"
        if [[ "${test_type}" == "e2e_c" || "${test_type}" == "e2e_d" || "${test_type}" == "e2e_aclgraph" ]]; then
            result_csv=$(${runner_func} "${op_name}" "${test_csv}" "${ops_test_path}" "${test_type#e2e_}")
        else
            result_csv=$(${runner_func} "${op_name}" "${test_csv}" "${ops_test_path}")
        fi
        test_ret=$?

        if [[ ${test_ret} -ne 0 ]]; then
            op_error_flag=1
        fi

        if [[ -n "${result_csv}" ]]; then
            result_csvs+=("${result_csv}")
            # Append result to the per-type array dynamically, e.g. kernel_csvs.
            # e2e_c/e2e_d share the e2e summary bucket.
            local result_bucket="${test_type%%_*}"
            eval "${result_bucket}_csvs+=(\"\${result_csv}\")"
        fi
    done

    summarize_op_results "${op_name}" "kernel" "${kernel_csvs[*]}"
    summarize_op_results "${op_name}" "aclnn" "${aclnn_csvs[*]}"
    summarize_op_results "${op_name}" "e2e" "${e2e_csvs[*]}"

    # Emit lines in 'op_name:testcase_name:result_csv' format for the caller
    for csv in "${result_csvs[@]}"; do
        testcase_name=$(basename "${csv}" _result.csv)
        echo "${op_name}:${testcase_name}:${csv}"
    done

    if [[ ${op_error_flag} -ne 0 ]]; then
        return 1
    fi
}

# Parse command-line arguments and validate inputs
parse_args() {
    ops_list=""
    soc_version=""
    test_type_list=""
    pr_filelist=""
    case_path=""
    testcase_filter=""
    update_ttk=""
    ci_st_check=""

    for arg in "$@"; do
        case "${arg}" in
            --ops=*)
                ops_list="${arg#*=}"
                ;;
            --soc_version=*)
                soc_version="${arg#*=}"
                ;;
            --test_type=*)
                test_type_list="${arg#*=}"
                ;;
            --pr_filelist=*)
                pr_filelist="${arg#*=}"
                ;;
            --case_path=*)
                case_path="${arg#*=}"
                ;;
            --testcase=*|-t=*)
                if [[ -n "${testcase_filter}" ]]; then
                    print_error "Cannot specify --testcase and -t together (or repeat either one)"
                    usage
                    exit 1
                fi
                testcase_filter="${arg#*=}"
                ;;
            --update_ttk)
                update_ttk="TRUE"
                ;;
            --ci_st_check)
                ci_st_check="TRUE"
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                print_error "Unknown argument: ${arg}"
                usage
                exit 1
                ;;
            *)
                if [[ -z "${pr_filelist}" ]]; then
                    pr_filelist="${arg}"
                else
                    print_error "Multiple pr_filelist arguments: ${pr_filelist} and ${arg}"
                    usage
                    exit 1
                fi
                ;;
        esac
    done

    if [[ -n "${ci_st_check}" ]]; then
        if [[ -n "${ops_list}${soc_version}${test_type_list}${case_path}${testcase_filter}${update_ttk}" ]]; then
            print_error "--ci_st_check can only be used with --pr_filelist; --ops/--soc_version/--test_type/--case_path/--testcase/--update_ttk are not allowed"
            usage
            exit 1
        fi
        if [[ -z "${pr_filelist}" ]]; then
            print_error "--ci_st_check must be used together with --pr_filelist=<file>"
            usage
            exit 1
        fi
    fi

    if [[ -n "${pr_filelist}" && ! -f "${pr_filelist}" && -z "${ci_st_check}" ]]; then
        print_error "pr_filelist not found: ${pr_filelist}"
        exit 1
    fi

    if [[ -n "${test_type_list}" ]]; then
        IFS=',' read -r -a valid_types <<< "kernel,aclnn,e2e"
        IFS=',' read -r -a input_types <<< "${test_type_list}"
        for input_type in "${input_types[@]}"; do
            local found=0
            for valid_type in "${valid_types[@]}"; do
                if [[ "${input_type}" == "${valid_type}" ]]; then
                    found=1
                    break
                fi
            done
            if [[ ${found} -eq 0 ]]; then
                print_error "Unsupported test_type: ${input_type}. Supported: kernel, aclnn, e2e"
                exit 1
            fi
        done
    fi

    print_msg "ops_list: ${ops_list:-'auto detect from git diff or pr_filelist'}"
    print_msg "test_type_list: ${test_type_list:-'all types'}"
    if [[ -n "${pr_filelist}" ]]; then
        print_msg "pr_filelist: ${pr_filelist}"
    fi
    if [[ -n "${case_path}" ]]; then
        print_msg "case_path: ${case_path}"
    fi
    if [[ -n "${update_ttk}" ]]; then
        print_msg "update_ttk: ${update_ttk}"
    fi
    if [[ -n "${ci_st_check}" ]]; then
        print_msg "ci_st_check: TRUE"
    fi
}

# Global paths used across functions
framework_path="$(cd "$(dirname "$0")/../.." && pwd)"
build_path="${framework_path}/build"
log_path="${framework_path}/st/log"
ttk_path="${build_path}/third_party/ops-test-kit"

parse_args "$@"

# Normalize pr_filelist to absolute path
if [[ -n "${pr_filelist}" && "${pr_filelist}" != /* ]]; then
    pr_filelist="$(pwd)/${pr_filelist}"
fi

# Normalize case_path to absolute path. Keep a missing relative path untouched;
# reproduction printing guards the find to avoid set -e aborts.
if [[ -n "${case_path}" ]]; then
    if [[ -d "${case_path}" ]]; then
        case_path="$(cd "${case_path}" && pwd)"
    else
        print_warning "case_path not found: ${case_path}"
    fi
fi

# Validate the optional graph process count; invalid values fall back to ttk default.
if [[ -n "${E2E_GRAPH_PC:-}" && ! "${E2E_GRAPH_PC}" =~ ^[1-9][0-9]*$ ]]; then
    print_warning "Invalid E2E_GRAPH_PC=${E2E_GRAPH_PC}; using ttk default"
    E2E_GRAPH_PC=""
fi

if [[ -n "${ci_st_check}" ]]; then
    run_ci_st_check "${pr_filelist}"
    exit 0
fi

rm -rf "${log_path:?}"/*
mkdir -p "${log_path}"

# Stage 1: ensure ops-test-kit is available
download_ops_test_kit

if [[ "${update_ttk}" == "TRUE" ]]; then
    print_success "ops-test-kit updated successfully."
    exit 0
fi

# Stage 2: source CANN environment (required by ttk info and tests)
setup_cann_env

# Stage 3: detect SOC version via ttk info
detect_soc_version

# Stage 4: determine ops to test (--ops, pr_filelist or git diff)
if [[ -n "${ops_list}" && -z "${pr_filelist}" ]]; then
    print_msg "Using ops from --ops parameter: ${ops_list}"
elif [[ -z "${ops_list}" && -n "${pr_filelist}" ]]; then
    print_msg "Extracting ops from pr_filelist..."
    print_msg "pr_filelist content:"
    cat "${pr_filelist}" | grep -v '^$' | grep -v '^#' | sed 's/^[MADRC]\t//' >&2
    ops_list=$(parse_ops_from_filelist "${pr_filelist}")
elif [[ -n "${ops_list}" && -n "${pr_filelist}" ]]; then
    print_msg "Merging ops from pr_filelist and --ops parameter..."
    print_msg "--ops input: ${ops_list}"
    print_msg "pr_filelist content:"
    cat "${pr_filelist}" | grep -v '^$' | grep -v '^#' | sed 's/^[MADRC]\t//' >&2
    ops_from_filelist=$(parse_ops_from_filelist "${pr_filelist}")
    ops_list=$(merge_ops_lists "${ops_from_filelist}" "${ops_list}")
else
    print_msg "Extracting ops from git diff..."
    ops_list=$(get_changed_ops)
    ops_list=$(echo "${ops_list}" | tr ';' ',')
fi

# Each package only contains its own operators: the mainline distest package has
# no experimental ops and the experimental package has no mainline ops.
st_op_scope="${ST_OP_SCOPE:-mainline}"
if [[ "${st_op_scope}" == "experimental" ]]; then
    st_scope_experimental="yes"
else
    st_scope_experimental="no"
fi
if [[ -n "${ops_list}" ]]; then
    ops_kept=""
    ops_dropped=""
    for op_candidate in ${ops_list//,/ }; do
        if [[ "${op_candidate}" == experimental/* ]]; then
            op_is_experimental="yes"
        else
            op_is_experimental="no"
        fi
        if [[ "${op_is_experimental}" == "${st_scope_experimental}" ]]; then
            ops_kept="${ops_kept:+${ops_kept},}${op_candidate}"
        else
            ops_dropped="${ops_dropped:+${ops_dropped},}${op_candidate}"
        fi
    done
    if [[ -n "${ops_dropped}" ]]; then
        print_msg "op scope ${st_op_scope}: skip ${ops_dropped}"
    fi
    ops_list="${ops_kept}"
fi

if [[ -z "${ops_list}" ]]; then
    print_msg "No ops to test"
    exit 0
fi

print_msg "Ops to test: ${ops_list}"

IFS=',' read -r -a op_name_array <<< "${ops_list}"

# Stage 4.5: install python deps that the selected operators' tests import.
op_code_dirs=()
for op_name in "${op_name_array[@]}"; do
    op_code_dir=$(find_op_code_path "${op_name}" 2>/dev/null) || continue
    op_code_dirs+=("${op_code_dir}")
done
ensure_op_python_deps ${op_code_dirs[@]+"${op_code_dirs[@]}"}

# Stage 5: run tests for each op
init_case_budget
all_result_csvs=()
result_flag=0
for op_name in "${op_name_array[@]}"; do
    op_results=$(run_single_op_test "${op_name}") || result_flag=1
    if [[ -n "${op_results}" ]]; then
        while IFS= read -r line; do
            # Only accept 'op:testcase:path' entries; anything else means the
            # function's stdout got polluted and must not feed the summary loop.
            if [[ "${line}" == *:*:* && "${line}" == *_result.csv ]]; then
                all_result_csvs+=("${line}")
            fi
        done <<< "${op_results}"
    fi
done

# Stage 6: precision check on all result CSVs
print_msg "=== Starting precision check for all test cases ==="
precision_flag=0
for result_info in "${all_result_csvs[@]}"; do
    op_name=$(echo "${result_info}" | cut -d':' -f1)
    testcase_name=$(echo "${result_info}" | cut -d':' -f2)
    result_csv=$(echo "${result_info}" | cut -d':' -f3)
    check_precision_status "${result_csv}" "${op_name}" "${testcase_name}" || precision_flag=1
done

print_summary_table

# Stage 7: print summary table and decide final exit code
if [[ ${result_flag} -ne 0 || ${precision_flag} -ne 0 ]]; then
    print_reproduction_commands
    print_error "Some tests or precision checks failed. See reproduction commands above."
    if [[ "${CI_ONLINE_MODE:-}" == "1" ]]; then
        print_warning "CI_ONLINE_MODE=1: ST failures are reported only, exiting with 0"
        exit 0
    fi
    exit 1
else
    print_success "All tests and precision checks passed."
    exit 0
fi
