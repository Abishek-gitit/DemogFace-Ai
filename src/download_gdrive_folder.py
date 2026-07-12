import os
import gdown

def main():
    folder_id = "1HROmgviy4jUUUaCdvvrQ8PcqtNg2jn3G"
    output_dir = "data/UTKFace_new"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading Google Drive folder {folder_id} into {output_dir} using gdown...")
    try:
        # Use gdown to download the folder contents
        gdown.download_folder(id=folder_id, output=output_dir, quiet=False, use_cookies=False)
        print("Download completed successfully.")
    except Exception as e:
        print(f"Error occurred during download: {e}")
        print("Please check if the folder is shared publicly.")

if __name__ == "__main__":
    main()
