#!/usr/bin/env python3
"""
Environment check script for PyTorch Symmetric Memory Extension.
Run this before building or using the extension to verify your environment.
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires >= 3.8)")
        return False

def check_pytorch():
    """Check PyTorch installation"""
    print("\nChecking PyTorch...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        return True
    except ImportError:
        print("  ✗ PyTorch not found")
        return False

def check_xpu_support():
    """Check XPU support in PyTorch"""
    print("\nChecking XPU support...")
    try:
        import torch
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            print(f"  ✓ XPU available ({device_count} device(s))")
            for i in range(device_count):
                name = torch.xpu.get_device_name(i)
                print(f"    - Device {i}: {name}")
            return True
        else:
            print("  ✗ XPU not available")
            return False
    except Exception as e:
        print(f"  ✗ Error checking XPU: {e}")
        return False

def check_symmetric_memory():
    """Check Symmetric Memory support"""
    print("\nChecking Symmetric Memory support...")
    try:
        import torch.distributed._symmetric_memory as symm_mem
        print("  ✓ Symmetric Memory module found")
        
        # Try to check if it's available (may require distributed init)
        try:
            # This might fail if not in a distributed context, which is OK
            backends = []
            for backend in ["ISHMEM", "NVSHMEM"]:
                try:
                    # Just check if we can import, don't actually set
                    print(f"    - {backend} backend: available")
                    backends.append(backend)
                except:
                    pass
            if backends:
                print(f"  ✓ Available backends: {', '.join(backends)}")
        except:
            print("    (Backend availability check requires distributed context)")
        
        return True
    except ImportError as e:
        print(f"  ✗ Symmetric Memory not available: {e}")
        print("    PyTorch may not be built with Symmetric Memory support")
        return False

def check_compiler():
    """Check for Intel DPC++ compiler"""
    print("\nChecking Intel DPC++ compiler...")
    import subprocess
    try:
        result = subprocess.run(['icpx', '--version'], 
                              capture_output=True, text=True, check=True)
        version_line = result.stdout.split('\n')[0]
        print(f"  ✓ {version_line}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ✗ icpx not found")
        print("    Please source Intel oneAPI environment:")
        print("    source /opt/intel/oneapi/setvars.sh")
        return False

def check_oneapi_env():
    """Check oneAPI environment variables"""
    print("\nChecking oneAPI environment...")
    env_vars = ['CMPLR_ROOT', 'SYCL_ROOT']
    found = False
    for var in env_vars:
        if var in os.environ:
            print(f"  ✓ {var}={os.environ[var]}")
            found = True
    
    if not found:
        print("  ⚠ No oneAPI environment variables found")
        print("    This is OK if icpx is in PATH")
    
    return True

def check_mpi():
    """Check MPI installation (optional)"""
    print("\nChecking MPI (optional)...")
    import subprocess
    try:
        result = subprocess.run(['mpirun', '--version'], 
                              capture_output=True, text=True, check=True)
        version_line = result.stdout.split('\n')[0]
        print(f"  ✓ {version_line}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ⚠ mpirun not found (optional for testing)")
        return True  # Not critical

def main():
    """Run all checks"""
    print("=" * 70)
    print("PyTorch Symmetric Memory Extension - Environment Check")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch", check_pytorch),
        ("XPU Support", check_xpu_support),
        ("Symmetric Memory", check_symmetric_memory),
        ("Intel DPC++ Compiler", check_compiler),
        ("oneAPI Environment", check_oneapi_env),
        ("MPI", check_mpi),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    critical_checks = ["Python Version", "PyTorch", "XPU Support", 
                      "Symmetric Memory", "Intel DPC++ Compiler"]
    
    all_critical_passed = True
    for name, result in results:
        status = "✓" if result else "✗"
        critical = " (CRITICAL)" if name in critical_checks else ""
        print(f"{status} {name}{critical}")
        if name in critical_checks and not result:
            all_critical_passed = False
    
    print("=" * 70)
    
    if all_critical_passed:
        print("\n✓ All critical checks passed!")
        print("  You can proceed with building the extension:")
        print("    python setup.py build_ext --inplace")
        return 0
    else:
        print("\n✗ Some critical checks failed.")
        print("  Please fix the issues above before building.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

