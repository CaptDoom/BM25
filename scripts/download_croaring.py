import os
import urllib.request

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    urllib.request.urlretrieve(url, dest_path)
    print("Download completed.")

def main():
    cpp_dir = os.path.abspath("src/cpp")
    os.makedirs(cpp_dir, exist_ok=True)
    
    roaring_c_url = "https://github.com/RoaringBitmap/CRoaring/releases/latest/download/roaring.c"
    roaring_h_url = "https://github.com/RoaringBitmap/CRoaring/releases/latest/download/roaring.h"
    
    download_file(roaring_c_url, os.path.join(cpp_dir, "roaring.c"))
    download_file(roaring_h_url, os.path.join(cpp_dir, "roaring.h"))
    print("CRoaring files downloaded successfully.")

if __name__ == "__main__":
    main()
