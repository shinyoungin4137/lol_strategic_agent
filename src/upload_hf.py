import os
from huggingface_hub import HfApi

# 1. Insert your Hugging Face Write Token here
HF_TOKEN = "my_HF_TOKEN"

# 2. Your target repository ID
REPO_ID = "younginAI/mistral-lol-agent"

# 3. Path to the folder containing your 3 exact files:
# (adapter_config.json, adapter_model.safetensors, master_meta.csv)
LOCAL_FOLDER_PATH = "MY_LOCAL_FOLDER"


def upload_to_hf():
    print("🚀 Initializing Hugging Face API...")
    api = HfApi()

    try:
        print(f"📂 Target local directory: {LOCAL_FOLDER_PATH}")
        print(f"☁️ Target repository: {REPO_ID}")
        print("⏳ Uploading files... This might take a few minutes depending on the file size.")

        # This function automatically handles large files (LFS) bypassing web UI bugs
        api.upload_folder(
            folder_path=LOCAL_FOLDER_PATH,
            repo_id=REPO_ID,
            repo_type="model",
            token=HF_TOKEN,
            commit_message="Upload actual model weights and CSV via Python API"
        )

        print("✅ Upload completed successfully!")
        print(f"🔗 Check your files at: https://huggingface.co/{REPO_ID}/tree/main")

    except Exception as e:
        print(f"❌ Upload failed. Error details:\n{e}")


if __name__ == "__main__":
    # Ensure the directory exists before attempting upload
    if not os.path.exists(LOCAL_FOLDER_PATH):
        print("❌ Error: The specified local folder path does not exist. Please check the path.")
    else:
        upload_to_hf()