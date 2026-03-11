import os
import subprocess
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, include_paths

# Force use of icpx compiler for SYCL support
os.environ['CXX'] = 'icpx'
os.environ['CC'] = 'icx'

REPO_ROOT = Path(__file__).resolve().parent
DEVICE_WRAPPER_SOURCE = REPO_ROOT / 'ishmem_wrapper.cpp'
DEVICE_WRAPPER_BC = REPO_ROOT / 'libtorch_ishmem_device.bc'
HOST_EXTENSION_SOURCES = ['ishmem_extension.cpp', 'binding.cpp']

# Get environment variables
ishmem_root = os.environ.get('ISHMEM_ROOT')
if not ishmem_root:
    raise RuntimeError("ISHMEM_ROOT environment variable not set")

sycl_root = os.environ.get('CMPLR_ROOT') or os.environ.get('SYCL_ROOT')
if not sycl_root:
    # Try to find from icpx
    try:
        result = subprocess.run(['which', 'icpx'], capture_output=True, text=True, check=True)
        icpx_path = result.stdout.strip()
        # icpx is usually in <SYCL_ROOT>/bin/icpx
        sycl_root = os.path.dirname(os.path.dirname(icpx_path))
    except:
        raise RuntimeError("Cannot find SYCL compiler root. Set CMPLR_ROOT or SYCL_ROOT")

# Include directories
common_include_dirs = [
    os.path.join(ishmem_root, 'include'),
    os.path.join(sycl_root, 'include'),
    os.path.join(sycl_root, 'include', 'sycl'),
]
host_include_dirs = [*common_include_dirs, *include_paths()]
device_include_dirs = [*common_include_dirs]

# Library directories and libraries
library_dirs = [
    os.path.join(ishmem_root, 'lib'),
    os.path.join(sycl_root, 'lib'),
]

# Don't add ishmem to libraries list since we use --whole-archive
libraries = ['sycl', 'ze_loader']

# Compiler flags for SYCL and ISHMEM device code
extra_compile_args = [
    '-fsycl',
    '-fsycl-targets=spir64,spir64_gen',
    '-fsycl-device-code-split=off',  # Critical for ISHMEM device linking
    '-fsycl-allow-device-image-dependencies',  # Allow external device symbols
    '-std=c++17',
    '-Wall',
    '-Wno-unused-parameter',
]

device_bc_compile_args = [
    '-fsycl',
    '-fsycl-device-only',
    '-fsycl-targets=spir64',
    '-fsycl-device-code-split=off',
    '-emit-llvm',
    '-std=c++17',
    '-Wall',
    '-Wno-unused-parameter',
]

# Linker flags
# Use --whole-archive for ISHMEM to ensure all device symbols are included
# Add -Wl,--allow-multiple-definition to handle potential duplicate weak symbols
# Specify device for AOT compilation
aot_device = os.environ.get('ISHMEM_AOT_DEVICE', 'pvc')  # Default to PVC
extra_link_args = [
    '-fsycl',
    '-fsycl-targets=spir64,spir64_gen',
    '-fsycl-device-code-split=off',
    '-Wl,--allow-multiple-definition',  # Allow duplicate weak symbols
    '-Xs', f'-device {aot_device}',  # Specify device for AOT
    f'-Wl,--whole-archive,{os.path.join(ishmem_root, "lib", "libishmem.a")},--no-whole-archive',
    '-lze_loader',
]

# Check if MPI is needed (if ISHMEM was built with MPI)
mpi_root = os.environ.get('MPI_ROOT') or os.environ.get('I_MPI_ROOT')
if mpi_root:
    mpi_include_dir = os.path.join(mpi_root, 'include')
    host_include_dirs.append(mpi_include_dir)
    device_include_dirs.append(mpi_include_dir)
    library_dirs.append(os.path.join(mpi_root, 'lib'))
    libraries.append('mpi')
    print(f"Using MPI from: {mpi_root}")


BaseBuildExtension = BuildExtension.with_options(use_ninja=False)


class BuildExtensionWithDeviceBc(BaseBuildExtension):
    def run(self):
        super().run()
        self.build_device_wrapper_bc()

    def build_device_wrapper_bc(self) -> None:
        compiler = os.environ.get('CXX', 'icpx')
        compile_cmd = [
            compiler,
            *device_bc_compile_args,
            '-c',
            str(DEVICE_WRAPPER_SOURCE),
            '-o',
            str(DEVICE_WRAPPER_BC),
        ]

        for include_dir in device_include_dirs:
            compile_cmd.extend(['-I', include_dir])

        print(f"Building Triton device wrapper bitcode: {DEVICE_WRAPPER_BC}")
        subprocess.run(compile_cmd, check=True)

setup(
    name='torch_ishmem_extension',
    version='0.1.0',
    description='Intel SHMEM extension for PyTorch XPU',
    ext_modules=[
        CppExtension(
            name='torch_ishmem_extension',
            sources=HOST_EXTENSION_SOURCES,
            include_dirs=host_include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language='c++',
        ),
    ],
    cmdclass={
        'build_ext': BuildExtensionWithDeviceBc
    },
    python_requires='>=3.8',
    install_requires=['torch'],
)
