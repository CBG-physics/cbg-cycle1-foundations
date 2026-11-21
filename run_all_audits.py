#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Master runner for the CBG Cycle I Foundational Trilogy verification suites."""

import glob
import os
import subprocess
import sys

FOLDERS = [
    "Paper_0_Principia_Geometrica_v4",
    "Paper_I_Foundations_v10",
    "Paper_II_Black_Hole_Mechanics_v10",
]


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    scripts = []
    for folder in FOLDERS:
        found = sorted(glob.glob(os.path.join(root, folder, "*.py")))
        scripts.extend(found)

    if not scripts:
        print("ERROR: No verification .py scripts found in Paper_* folders.")
        sys.exit(1)

    for script in scripts:
        rel = os.path.relpath(script, root)
        print("\n" + "#" * 94)
        print(f"RUNNING SUITE: {rel}")
        print("#" * 94)
        res = subprocess.run([sys.executable, script], cwd=os.path.dirname(script))
        if res.returncode != 0:
            print(f"\n[FAILED] {rel} exited with code {res.returncode}")
            sys.exit(res.returncode)

    print("\n" + "=" * 94)
    print(f"ALL {len(scripts)} CBG CYCLE I VERIFICATION SUITES PASSED WITH ZERO FAILURES.")
    print("=" * 94)


if __name__ == "__main__":
    main()
