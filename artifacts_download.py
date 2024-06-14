import os
import zipfile
import wandb
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Initialize a W&B run
wandb.init(project="whisper_artifacts_upload")

# Create a directory to store artifacts
artifacts_dir = "artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

# Download and save the processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
processor.save_pretrained(artifacts_dir)

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").to("cuda")
model.save_pretrained(artifacts_dir)

# Zip the artifacts directory
zip_filename = "asr_tiny_base.zip"
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for root, dirs, files in os.walk(artifacts_dir):
        for file in files:
            zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), artifacts_dir))

# Upload the zip file to W&B
artifact = wandb.Artifact('whisper_artifacts', type='model')
artifact.add_file(zip_filename)
wandb.log_artifact(artifact)

# Load the processor and model from the zip file for inference
with zipfile.ZipFile(zip_filename, 'r') as zipf:
    zipf.extractall(artifacts_dir)

processor = WhisperProcessor.from_pretrained(artifacts_dir)
model = WhisperForConditionalGeneration.from_pretrained(artifacts_dir).to("cuda")
