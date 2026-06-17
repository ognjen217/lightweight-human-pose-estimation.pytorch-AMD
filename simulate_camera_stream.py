#!/usr/bin/env python3
"""Root wrapper for the modular multi-camera stream simulator."""

from simulation.split_hip_fused_patch import apply_patch

apply_patch()

from simulation.cli import main


if __name__ == "__main__":
    main()
