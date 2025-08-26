#Command to run the FastAPI server: python -m uvicorn scripts.api_server:app --reload
# filepath: [api_server.py](http://_vscodecontentref_/2)
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from moviepy.editor import AudioFileClip
import imageio_ffmpeg as ffmpeg_binaries
import ffmpeg
from scripts.trainConformer import ConformerModel, BarkVoiceCommandDataset
import torch
import torchaudio

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and class names once at startup
NUM_CLASSES = 3
MODEL_PATH = "models/conformer_best_model.pth"
DATASET_PATH = "data_barkAI_large"
model = ConformerModel(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
dataset = BarkVoiceCommandDataset(DATASET_PATH, mode='val')
class_names = dataset.unique_labels

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    # Save the uploaded file
    save_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Convert the file to WAV format
    wav_file_path = f"uploads/{os.path.splitext(file.filename)[0]}.wav"
    convert_to_wav(save_path, wav_file_path)
    os.remove(save_path)

    # Preprocess audio
    waveform, sr = torchaudio.load(wav_file_path)
    waveform = waveform.mean(dim=0, keepdim=True) 
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=64)(waveform)
    mel_spec = mel_spec.unsqueeze(0) 

    with torch.no_grad():
        outputs = model(mel_spec)
        _, predicted = torch.max(outputs, 1)
        pred_idx = predicted.item()
        print("Predicted index:", pred_idx)
        print("Class names:", class_names)
        if 0 <= pred_idx < len(class_names):
            predicted_command = class_names[pred_idx]
        else:
            predicted_command = "Unknown command"
        print("Predicted command:", predicted_command)

    return {"predicted_command": predicted_command}

def convert_to_wav(input_file, output_file):
    try:
        ffmpeg_path = ffmpeg_binaries.get_ffmpeg_exe()
        (
            ffmpeg
            .input(input_file)
            .output(output_file, format='wav', acodec='pcm_s16le', ac=1, ar='44100')
            .run(cmd=ffmpeg_path, overwrite_output=True)
        )
        print(f"Converted {input_file} to {output_file}")
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")