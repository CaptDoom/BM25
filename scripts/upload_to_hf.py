#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AuraAI Index Uploader to Hugging Face Hub
Uploads the local FEVER index directory to a Hugging Face Dataset repository.
"""
import os
import argparse
from huggingface_hub import HfApi, login

def main():
    parser = argparse.ArgumentParser(description="Upload AuraAI Index to Hugging Face Dataset Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face Dataset repo ID (e.g., username/repo-name)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token. If not provided, you must run 'huggingface-cli login' first or we will prompt you.")
    args = parser.parse_args()

    local_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "index", "BeIR_fever"))
    
    if not os.path.exists(local_folder):
        print(f"[!] Local index directory not found at: {local_folder}")
        print("Please build the FEVER index locally first.")
        return

    # Log in if token is provided
    if args.token:
        print("[*] Logging in to Hugging Face with provided token...")
        login(token=args.token)
    else:
        print("[*] Using cached credentials or prompting for login...")

    print(f"[*] Preparing to upload local folder: {local_folder}")
    print(f"[*] Target Hugging Face Repository: {args.repo_id} (Dataset)")

    try:
        api = HfApi()
        
        # Ensure repository exists
        print("[*] Creating/Verifying target repository on Hugging Face Hub...")
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
        
        # Upload folder
        print("[*] Uploading index folder... This may take several minutes as the database is 3.5GB+. Please do not interrupt.")
        future = api.upload_folder(
            folder_path=local_folder,
            path_in_repo="BeIR_fever",
            repo_id=args.repo_id,
            repo_type="dataset"
        )
        print(f"[+] Success! Files uploaded to: https://huggingface.co/datasets/{args.repo_id}")
        
    except Exception as e:
        print(f"[!] Upload failed with error: {e}")
        print("Please ensure your token has 'write' permissions and the repository name is correct.")

if __name__ == "__main__":
    main()
