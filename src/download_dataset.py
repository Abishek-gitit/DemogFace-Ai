import os
import tarfile
import gdown

def main():
    target_dir = "data"
    os.makedirs(target_dir, exist_ok=True)
    
    tar_path = os.path.join(target_dir, "UTKFace.tar.gz")
    extract_path = os.path.join(target_dir, "UTKFace")
    
    if os.path.exists(extract_path):
        print(f"Dataset already extracted at {extract_path}")
        return
        
    if not os.path.exists(tar_path):
        print("Downloading UTKFace.tar.gz via gdown...")
        file_id = "0BxYys69jI14kYVM3aVhKS1VhRUk"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            gdown.download(url, tar_path, quiet=False)
        except Exception as e:
            print(f"Error downloading via gdown: {e}")
            print("Please download manually from: https://susanqq.github.io/UTKFace/")
            print(f"Place the 'UTKFace.tar.gz' file under the '{target_dir}' folder.")
            return

    print("Extracting UTKFace.tar.gz...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
        print(f"Successfully extracted to {extract_path}")
    except Exception as e:
        print(f"Error extracting: {e}")

if __name__ == "__main__":
    main()
