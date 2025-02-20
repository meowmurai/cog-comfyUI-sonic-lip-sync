# Predict
- python ./scripts/reset.py
- python ./scripts/download_custom_weights.py (need download manually also)
- cog build
- cog predict -i input_image=@./anime1.png -i input_audio=@./sing_female_10s.wav
- uncommont /ComfyUI from .gitignore
- cog push
