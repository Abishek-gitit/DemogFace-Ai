import os
import tarfile

def extract_tar(tar_path, extract_dir):
    print(f"Extracting {tar_path} to {extract_dir}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print(f"Finished extracting {tar_path}")
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")

def main():
    data_dir = "data/UTKFace_new"
    archives = ["part1.tar.gz", "part2.tar.gz", "part3.tar.gz"]
    
    for archive in archives:
        tar_path = os.path.join(data_dir, archive)
        if os.path.exists(tar_path):
            extract_tar(tar_path, data_dir)
        else:
            print(f"Archive not found: {tar_path}")

if __name__ == "__main__":
    main()
