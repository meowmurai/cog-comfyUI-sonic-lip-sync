# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    # Update nodes in the JSON workflow to modify your workflow based on the given inputs
    def update_workflow(self, workflow, **kwargs):
        # Below is an example showing how to get the node you need and update the inputs

        load_audio = workflow["10"]["inputs"]
        load_audio["audio"] = kwargs["audio_filename"]

        load_image = workflow["2"]["inputs"]
        load_image["image"] = kwargs["image_filename"]

        sampler = workflow["4"]["inputs"]
        sampler["seed"] = kwargs["seed"]

        video_combiner = workflow["5"]["inputs"]
        video_combiner["format"] = kwargs["output_format"]

    def predict(
        self,
        input_image: Path = Input(
            description="Input image",
            default=None,
        ),
        input_audio: Path = Input(
            description="Input audio",
            default=None
        ),
        output_format: str = Input(
            description="Output's format, ex: image/gif, image/webp, video/h264-mp4, ...",
            default="image/gif"
        ),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # Make sure to set the seeds in your workflow
        seed = seed_helper.generate(seed)

        image_filename = None
        if input_image:
            image_filename = self.filename_with_extension(input_image, "image")
            self.handle_input_file(input_image, image_filename)

        if input_audio:
            audio_filename = self.filename_with_extension(input_image, "audio")
            self.handle_input_file(input_image, audio_filename)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            image_filename=image_filename,
            audio_filename=audio_filename,
            output_format=output_format,
            seed=seed,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        # return optimise_images.optimise_image_files(
        #     output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        # )
        return self.comfyUI.get_files(COMFYUI_TEMP_OUTPUT_DIR)
