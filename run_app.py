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
    
    # Launch Streamlit
    print("[*] Launching AuraAI...")
    print("[+] Opening browser at http://localhost:8501\n")
    
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, 
           "--logger.level=warning",
           "--client.showErrorDetails=false",
           "--client.toolbarMode=minimal"]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
