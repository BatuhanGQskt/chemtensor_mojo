#!/bin/bash
set -e

# Parse arguments: -dw/--disable-warnings, --quiet, and optional test dir (last arg).
# Usage: run_tests.sh [-dw|--disable-warnings] [--quiet] [test_dir]
# Example: bash tools/run_tests.sh -dw src/tests/algorithms
MOJO_EXTRA_FLAGS=""
QUIET=false
test_dir="src/tests"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -dw|--disable-warnings)
      MOJO_EXTRA_FLAGS="$MOJO_EXTRA_FLAGS --disable-warnings"
      shift
      ;;
    -q|--quiet)
      QUIET=true
      shift
      ;;
    *)
      test_dir="$1"
      shift
      break
      ;;
  esac
done
any_failed=false

# Arrays for summary: one entry per test file (same index)
TEST_FILES=()
TEST_PASSED=()   # 1 = passed, 0 = failed
TEST_FUNCTIONS=() # newline-separated "  -- fn_name passed|failed" (empty if not TestSuite output)

echo "Running tests in $test_dir..."

# Check if mojo is in path
if ! command -v mojo &> /dev/null; then
    echo "Error: 'mojo' command not found."
    echo "Please ensure Mojo is installed and in your PATH,"
    echo "or run this script within a 'pixi shell'."
    exit 1
fi

# Helper: parse Mojo TestSuite output for "PASS [ ... ] fn_name" and "FAIL [ ... ] fn_name"
parse_test_suite_output() {
  local out_file="$1"
  local lines=""
  while IFS= read -r line; do
    if [[ "$line" =~ ^(PASS|FAIL)[[:space:]]+\[.*\][[:space:]]+(.+)$ ]]; then
      local status="${BASH_REMATCH[1]}"
      local fn="${BASH_REMATCH[2]}"
      if [[ "$status" == "PASS" ]]; then
        lines="${lines}${lines:+$'\n'}  -- ${fn} passed"
      else
        lines="${lines}${lines:+$'\n'}  -- ${fn} failed"
      fi
    fi
  done < "$out_file"
  echo "$lines"
}

# Find all test_*.mojo files
while IFS= read -r test_file; do
  echo "----------------------------------------------------------------"
  echo "Running: $test_file"
  echo "----------------------------------------------------------------"

  OUTPUT_FILE=$(mktemp)
  set +e
  mojo run -I . $MOJO_EXTRA_FLAGS "$test_file" > "$OUTPUT_FILE" 2>&1
  exit_code=$?
  set -e

  if [ $exit_code -ne 0 ]; then
    if [ "$QUIET" != true ]; then
      cat "$OUTPUT_FILE"
    fi
    echo "FAILED: $test_file (exit code $exit_code)"
    any_failed=true
    TEST_FILES+=("$test_file")
    TEST_PASSED+=(0)
    TEST_FUNCTIONS+=("$(parse_test_suite_output "$OUTPUT_FILE")")
  else
    if [ "$QUIET" != true ]; then
      cat "$OUTPUT_FILE"
    fi
    echo "PASSED: $test_file"
    TEST_FILES+=("$test_file")
    TEST_PASSED+=(1)
    TEST_FUNCTIONS+=("$(parse_test_suite_output "$OUTPUT_FILE")")
  fi
  rm -f "$OUTPUT_FILE"
done < <(find "$test_dir" -name "test_*.mojo" -type f | sort)

# --- Summary ---
echo ""
echo "========================================================================"
echo "                            TEST SUMMARY"
echo "========================================================================"

# Group by directory: dir -> space-separated list of test file paths (same order as TEST_FILES for that dir)
declare -A dir_file_list
for i in "${!TEST_FILES[@]}"; do
  f="${TEST_FILES[$i]}"
  rel="${f#$test_dir/}"
  dir="${rel%/*}"
  if [[ "$dir" == "$rel" ]]; then
    dir="."
  fi
  if [[ -z "${dir_file_list[$dir]:-}" ]]; then
    dir_file_list[$dir]="$f"
  else
    dir_file_list[$dir]="${dir_file_list[$dir]} $f"
  fi
done

# Find index in TEST_FILES for a given path
find_index() {
  local path="$1"
  for i in "${!TEST_FILES[@]}"; do
    if [[ "${TEST_FILES[$i]}" == "$path" ]]; then
      echo "$i"
      return
    fi
  done
}

total_run=${#TEST_FILES[@]}
total_passed=0
for i in "${TEST_PASSED[@]}"; do total_passed=$((total_passed + i)); done
total_failed=$((total_run - total_passed))

sorted_dirs=$(echo "${!dir_file_list[@]}" | tr ' ' '\n' | sort)
while IFS= read -r dir; do
  [[ -z "$dir" ]] && continue
  files_str="${dir_file_list[$dir]}"
  dir_label="$test_dir/$dir"
  [[ "$dir" == "." ]] && dir_label="$test_dir"
  n=0
  passed_in_dir=0
  for f in $files_str; do
    n=$((n + 1))
    idx=$(find_index "$f")
    passed_in_dir=$((passed_in_dir + TEST_PASSED[idx]))
  done
  echo ""
  echo "$dir_label/ had $n test(s) ($passed_in_dir passed, $((n - passed_in_dir)) failed)."
  for f in $files_str; do
    idx=$(find_index "$f")
    status="PASSED"
    [[ "${TEST_PASSED[$idx]}" -eq 0 ]] && status="FAILED"
    basename_f="${f##*/}"
    echo "  - $basename_f  [$status]"
    fn_lines="${TEST_FUNCTIONS[$idx]}"
    if [[ -n "$fn_lines" ]]; then
      echo "$fn_lines"
    fi
  done
done <<< "$sorted_dirs"

echo ""
echo "----------------------------------------------------------------------"
echo "Total: $total_run test file(s) — $total_passed passed, $total_failed failed"
echo "========================================================================"

if [ "$any_failed" = true ]; then
  exit 1
else
  exit 0
fi
