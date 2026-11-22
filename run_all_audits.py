#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Master runner for the CBG Cycle I Foundational Monograph verification suites."""

import glob
import os
import subprocess
import sys

# Clean unversioned folder names (with legacy fallback during folder migration)
PAPER_GROUPS = [
    ("Paper 0", ["Paper_0_Principia_Geometrica", "Paper_0_Principia_Geometrica_v4"]),
    ("Paper I", ["Paper_I_Foundations", "Paper_I_Foundations_v10"]),
    ("Paper II", ["Paper_II_Black_Hole_Mechanics", "Paper_II_Black_Hole_Mechanics_v10"]),
    ("Paper III", ["Paper_III_Cosmology"]),
]


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    scripts = []

    for label, candidates in PAPER_GROUPS:
        matched_folder = None
        for folder in candidates:
            full_dir = os.path.join(root, folder)
            if os.path.isdir(full_dir):
                matched_folder = full_dir
                break

        if matched_folder is None:
            print(f"ERROR: Folder for {label} not found (checked: {candidates}).")
            sys.exit(1)

        found = sorted(glob.glob(os.path.join(matched_folder, "*.py")))
        if not found:
            print(f"ERROR: No verification .py script found inside {os.path.relpath(matched_folder, root)}/.")
            sys.exit(1)

        scripts.extend(found)

    for script in scripts:
        rel = os.path.relpath(script, root)
        print("\n" + "#" * 94)
        print(f"RUNNING SUITE: {rel}")
        print("#" * 94)
        res = subprocess.run([sys.executable, "-W", "error", script], cwd=os.path.dirname(script))
        if res.returncode != 0:
            print(f"\n[FAILED] {rel} exited with code {res.returncode}")
            sys.exit(res.returncode)

    print("\n" + "=" * 94)
    print(f"ALL {len(scripts)} CBG CYCLE I VERIFICATION SUITES PASSED WITH ZERO FAILURES.")
    print("=" * 94)


if __name__ == "__main__":
    main()
