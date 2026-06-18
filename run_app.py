#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AuraAI Application Launcher
Runs the Streamlit app with proper PYTHONPATH configuration
"""
import os
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Fix encoding for Windows console
    if sys.platform == "win32":
        os.environ["PYTHONIOENCODING"] = "utf-8"
    
    app_path = os.path.join(os.path.dirname(__file__), "src", "app.py")
    
    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Validate System Configuration on startup using Pydantic SystemConfig
    try:
        import yaml
        from src.config_schema import SystemConfig
        config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                raw_cfg = yaml.safe_load(f)
            # Parse configuration schema via Pydantic model
            SystemConfig(**raw_cfg)
            print("[*] Configuration validated successfully.")
        else:
            print("[Warning] No configuration file found to validate on startup.")
    except Exception as ve:
        print(f"[Error] Configuration validation failed: {ve}")
        sys.exit(1)

    
    # Launch Streamlit
    print("[*] Launching AuraAI...")
    print("[+] Opening browser at http://localhost:8501\n")
    
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, 
           "--logger.level=warning",
           "--client.showErrorDetails=false",
           "--client.toolbarMode=minimal"]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
