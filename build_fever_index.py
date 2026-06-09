#!/usr/bin/env python3
"""
FEVER Dataset Index Builder
Quick setup for BeIR/fever dataset indexing
"""
import os
import sys
import argparse

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def build_fever_index():
    """Build FEVER dataset index"""
    from src.evaluate import build_index
    
    config_path = "config/config.yaml"
    dataset_name = "BeIR/fever"
    index_dir = os.path.join("src/index", dataset_name.replace("/", "_"))
    
    print(f"""
🔥 Building FEVER Dataset Index
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset: {dataset_name}
Documents: ~5.4 million
Index Location: {index_dir}

This will take ~15-30 minutes depending on your system.
Starting indexing...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
    
    try:
        build_index(dataset_name, config_path, index_dir)
        print(f"\n✅ FEVER index built successfully!")
        print(f"   Location: {index_dir}")
        print(f"   The app should now find the index automatically.")
        return True
    except Exception as e:
        print(f"\n❌ Error building index: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Ensure you have 8GB+ RAM available")
        print(f"  2. Check your internet connection")
        print(f"  3. Try again with: python build_fever_index.py")
        return False

if __name__ == "__main__":
    success = build_fever_index()
    sys.exit(0 if success else 1)
