#!/usr/bin/env python3
"""
Launcher alias for OJT Journal Entity Extraction Studio GUI.
Calls tools.pipeline_gui.main().
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.pipeline_gui import main

if __name__ == "__main__":
    main()
