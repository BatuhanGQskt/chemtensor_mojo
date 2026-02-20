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
    --quiet)
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

echo "Running tests in $test_dir..."

# Check if mojo is in path
if ! command -v mojo &> /dev/null; then
    echo "Error: 'mojo' command not found."
    echo "Please ensure Mojo is installed and in your PATH,"
    echo "or run this script within a 'pixi shell'."
    exit 1
fi

# Find all test_*.mojo files
while IFS= read -r test_file; do
  echo "----------------------------------------------------------------"
  echo "Running: $test_file"
  echo "----------------------------------------------------------------"
  
  export CHEMTENSOR_DUMP_DIR=/home/batu/Master_Thesis/compare_chemtensors/out
  export CHEMTENSOR_ISING_J=1.0
  export CHEMTENSOR_ISING_H=0.0
  export CHEMTENSOR_ISING_G=1.0
  
  # Run the test file using 'mojo run'
  # We use 'set +e' temporarily to capture failure without exiting the script immediately
  # Capture output to a temporary file to suppress warnings on success
  OUTPUT_FILE=$(mktemp)
  set +e
  mojo run -I . $MOJO_EXTRA_FLAGS "$test_file" > "$OUTPUT_FILE" 2>&1
  exit_code=$?
  set -e
  
  if [ $exit_code -ne 0 ]; then
    cat "$OUTPUT_FILE"
    echo "FAILED: $test_file (exit code $exit_code)"
    any_failed=true
  else
    # Only print output if verbose or if explicitly requested (could add flag)
    # For now, suppressing output for passed tests to hide warnings
    cat "$OUTPUT_FILE" 
    echo "PASSED: $test_file"
  fi
  rm -f "$OUTPUT_FILE"
done < <(find "$test_dir" -name "test_*.mojo" -type f | sort)

if [ "$any_failed" = true ]; then
  echo "----------------------------------------------------------------"
  echo "SOME TESTS FAILED"
  exit 1
else
  echo "----------------------------------------------------------------"
  echo "ALL TESTS PASSED"
  exit 0
fi
