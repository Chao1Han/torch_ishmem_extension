import os
import sys
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension, include_paths
import torch

# Force use of icpx compiler for SYCL support
os.environ['CXX'] = 'icpx'
os.environ['CC'] = 'icx'

# Get environment variables
sycl_root = os.environ.get('CMPLR_ROOT') or os.environ.get('SYCL_ROOT')
if not sycl_root:
    # Try to find from icpx
    import subprocess
    try:
        result = subprocess.run(['which', 'icpx'], capture_output=True, text=True, check=True)
        icpx_path = result.stdout.strip()
        # icpx is usually in <SYCL_ROOT>/bin/icpx
        sycl_root = os.path.dirname(os.path.dirname(icpx_path))
    except:
        raise RuntimeError("Cannot find SYCL compiler root. Set CMPLR_ROOT or SYCL_ROOT")

# Include directories
include_dirs = [
    os.path.join(sycl_root, 'include'),
    os.path.join(sycl_root, 'include', 'sycl'),
]
include_dirs.extend(include_paths())

# Library directories and libraries
library_dirs = [
    os.path.join(sycl_root, 'lib'),
]

libraries = ['sycl', 'ze_loader']

# Compiler flags for SYCL device code
extra_compile_args = [
    '-fsycl',
    '-fsycl-targets=spir64,spir64_gen',
    '-std=c++17',
    '-Wall',
    '-Wno-unused-parameter',
]

# Linker flags
# Specify device for AOT compilation
aot_device = os.environ.get('AOT_DEVICE', 'pvc')  # Default to PVC
extra_link_args = [
    '-fsycl',
    '-fsycl-targets=spir64,spir64_gen',
    '-Xs', f'-device {aot_device}',  # Specify device for AOT
    '-lze_loader',
]

# Check if MPI is needed (if ISHMEM was built with MPI)
mpi_root = os.environ.get('MPI_ROOT') or os.environ.get('I_MPI_ROOT')
if mpi_root:
    include_dirs.append(os.path.join(mpi_root, 'include'))
    library_dirs.append(os.path.join(mpi_root, 'lib'))
    libraries.append('mpi')
    print(f"Using MPI from: {mpi_root}")

setup(
    name='torch_symm_mem_extension',
    version='0.2.0',
    description='PyTorch Symmetric Memory extension for XPU',
    ext_modules=[
        CppExtension(
            name='torch_symm_mem_extension',
            sources=['ishmem_extension.cpp', 'binding.cpp'],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language='c++',
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    },
    python_requires='>=3.8',
    install_requires=['torch'],
)
