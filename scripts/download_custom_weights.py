import gdown
from huggingface_hub import snapshot_download, login, hf_hub_download
import os
import time

login()

start_time = time.time()
comfy_model_folder = "./ComfyUI/models"
sonic_weight_url = "https://drive.google.com/drive/folders/1jI32B-2JX17seSGG0-MnZgUhCMHCEZlx"
gdown.download_folder(url=sonic_weight_url, output=os.path.join(comfy_model_folder, "sonic"))
elapsed_time = time.time() - start_time
print(
    f"✅ sonic checkpoints downloaded  in {elapsed_time:.2f}s"
)

start_time = time.time()
comfy_model_folder = "./ComfyUI/models"
sonic_weight_url = "https://drive.google.com/drive/folders/1QIIDvCDU-rp1ZB8qDA6NQqVn8F9WYMhE"
gdown.download_folder(url=sonic_weight_url, output=os.path.join(comfy_model_folder, "sonic/RIFE"))
elapsed_time = time.time() - start_time
print(
    f"✅ sonic checkpoints downloaded  in {elapsed_time:.2f}s"
)

start_time = time.time()
snapshot_download(repo_id="openai/whisper-tiny",allow_patterns=["model.safetensors", "preprocessor_config.json", "config.json"], local_dir=os.path.join(comfy_model_folder, "sonic/whisper-tiny"), local_dir_use_symlinks=False)
elapsed_time = time.time() - start_time
print(
    f"✅ whisper-tiny checkpoints downloaded  in {elapsed_time:.2f}s"
)

start_time = time.time()
hf_hub_download(repo_id="stabilityai/stable-video-diffusion-img2vid-xt-1-1", filename="svd_xt_1_1.safetensors", local_dir=os.path.join(comfy_model_folder, "checkpoints"))
elapsed_time = time.time() - start_time
print(
    f"✅ svd checkpoint downloaded  in {elapsed_time:.2f}s"
)
