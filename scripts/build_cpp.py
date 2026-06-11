import os
import subprocess
import sys

def compile_with_gpp(src_dir):
    dll_path = os.path.join(src_dir, "bm25_score.dll")
    cpp_src = os.path.join(src_dir, "bm25_score.cpp")
    c_src = os.path.join(src_dir, "roaring.c")
    
    cmd = [
        "g++",
        "-O3",
        "-shared",
        "-static",
        "-static-libgcc",
        "-static-libstdc++",
        "-std=c++17",
        "-o", dll_path,
        cpp_src,
        c_src
    ]
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Successfully compiled {dll_path} using g++.")
        return True
    else:
        print("g++ compilation failed:")
        print(result.stdout)
        print(result.stderr)
        return False

def compile_with_msvc(src_dir):
    dll_path = os.path.join(src_dir, "bm25_score.dll")
    cpp_src = os.path.join(src_dir, "bm25_score.cpp")
    c_src = os.path.join(src_dir, "roaring.c")
    
    # cl /O2 /LD /std:c++17 /Fe:src/cpp/bm25_score.dll src/cpp/bm25_score.cpp src/cpp/roaring.c
    cmd = [
        "cl",
        "/O2",
        "/LD",
        "/std:c++17",
        f"/Fe:{dll_path}",
        cpp_src,
        c_src
    ]
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Successfully compiled {dll_path} using MSVC cl.")
        return True
    else:
        print("MSVC cl compilation failed:")
        print(result.stdout)
        print(result.stderr)
        return False

def main():
    src_dir = os.path.abspath("src/cpp")
    if not os.path.exists(src_dir):
        print(f"Error: {src_dir} does not exist.")
        sys.exit(1)
        
    print("Attempting to compile C++ engine...")
    if compile_with_gpp(src_dir):
        print("Build complete!")
        return
        
    print("Falling back to MSVC cl...")
    if compile_with_msvc(src_dir):
        print("Build complete!")
        return
        
    print("Error: Could not compile C++ engine with g++ or cl.")
    print("The system will use the pure-Python fallback implementation at runtime.")
    # Exit with code 0 to allow the installation process to proceed, since we have a soft fallback
    sys.exit(0)

if __name__ == "__main__":
    main()
