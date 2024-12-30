import os
import subprocess
from packaging.version import parse, Version

from setuptools import setup

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    load,
)


def raise_if_cuda_home_none() -> None:
    if CUDA_HOME is not None:
        return
    raise RuntimeError(
        "nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def check_cupti_directory(cuda_dir):
    cupti_dir = os.path.join(cuda_dir, 'extras', 'CUPTI')
    if not os.path.isdir(cupti_dir):
        raise RuntimeError(
            f"CUPTI not found in {cupti_dir}. Please ensure that CUPTI is installed "
            "as part of your CUDA installation. CUPTI is required for profiling."
        )
    return cupti_dir


# Ensure CUDA_HOME is set and the directories are valid
raise_if_cuda_home_none()

# Check if CUPTI directory exists within CUDA_HOME
cupti_dir = check_cupti_directory(CUDA_HOME)

# Setup extension
setup(
    name='vtrain_profiler',
    ext_modules=[
        CppExtension(
            name='vtrain_profiler',
            sources=['cupti.cpp'],
            include_dirs=[
                os.path.join(cupti_dir, 'include'),
                os.path.join(CUDA_HOME, 'targets', 'x86_64-linux', 'include'),
            ],
            library_dirs=[
                os.path.join(CUDA_HOME, 'lib64'),
                os.path.join(cupti_dir, 'lib64'),
            ],
            libraries=['cupti']
            ),
        ],
    cmdclass={
        'build_ext': BuildExtension,
    }
)
