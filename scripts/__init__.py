"""
Hybrid NER + Classification Pipeline for OJT Journal Task Tagging.
Package initialization and GPU runtime environment bootstrap.
"""

import os
import sys
import glob
import ctypes
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("ojt_pipeline")


def _bootstrap_cuda_environment() -> None:
    """Preload NVIDIA CUDA runtime libraries to ensure seamless CuPy/spaCy GPU execution."""
    import site
    search_dirs = list(sys.path) + site.getsitepackages() + [
        os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    ]
    nvidia_lib_dirs = set()
    for sp in search_dirs:
        nvidia_dir = os.path.join(sp, "nvidia")
        if os.path.isdir(nvidia_dir):
            for lib_dir in glob.glob(os.path.join(nvidia_dir, "*", "lib")):
                nvidia_lib_dirs.add(lib_dir)
                for so_path in sorted(glob.glob(os.path.join(lib_dir, "*.so*"))):
                    try:
                        ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass

    if nvidia_lib_dirs:
        curr_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld = ":".join(nvidia_lib_dirs)
        os.environ["LD_LIBRARY_PATH"] = f"{new_ld}:{curr_ld}" if curr_ld else new_ld


_bootstrap_cuda_environment()


def init_gpu() -> bool:
    """Initializes spaCy GPU acceleration if available.
    
    Returns:
        bool: True if GPU is active, False otherwise.
    """
    try:
        import spacy
        import torch

        if torch.cuda.is_available():
            gpu_active = spacy.require_gpu()
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU accelerated with {device_name} (spacy.require_gpu={gpu_active})")
            return bool(gpu_active)
        else:
            logger.warning("CUDA not detected by PyTorch; falling back to CPU.")
            return False
    except Exception as e:
        logger.warning(f"Could not activate GPU for spaCy: {e}. Falling back to CPU.")
        return False
