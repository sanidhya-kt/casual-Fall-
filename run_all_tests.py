import os
import sys
import subprocess

test_files = [
    "tests/test_preprocessing.py",
    "tests/test_stae.py",
    "tests/test_cee.py",
    "tests/test_state_decoder.py",
    "tests/test_counterfactual.py",
    "tests/test_loss.py",
    "tests/test_causalfall_integration.py",
    "tests/test_baselines.py",
    "tests/test_pipeline_dryrun.py"
]

all_passed = True
print("=" * 60)
print("RUNNING ALL CAUSALFALL REPLICATION UNIT & INTEGRATION TESTS")
print("=" * 60)

for tfile in test_files:
    print(f"\n---> Executing {tfile} ...")
    result = subprocess.run([sys.executable, tfile], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[SUCCESS] {tfile}")
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if "[PASS]" in line or "PASSED" in line:
                    print("   ", line)
    else:
        print(f"[FAILURE] {tfile}")
        print(result.stdout)
        print(result.stderr)
        all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("ALL TESTS COMPLETED WITH 100% SUCCESS!")
else:
    print("SOME TESTS FAILED! CHECK OUTPUT ABOVE.")
print("=" * 60)
sys.exit(0 if all_passed else 1)
