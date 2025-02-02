# Description: Verify that all lowest level directories have exactly 7 files.
# Only used after using sorting_script.py

import os

def verify_subfolder(directory):
    """Check that all lowest level directories have exactly 7 files"""
    total_not_correct = 0
    for root, dirs, files in os.walk(directory):
        if not dirs:
            if len(files) != 7:
                print(f"Directory {root} does not contain 7 files.")
                print(f">   Files found: {len(files)}")
                total_not_correct += 1
    print(f"\nTotal directories not containing 7 files: {total_not_correct}")

verify_subfolder("data/data_sorted")