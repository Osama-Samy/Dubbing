import os
import tempfile
import shutil
import logging
from typing import List, Dict, Optional, Tuple
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import whisper
from transformers import MarianMTModel, MarianTokenizer
import azure.cognitiveservices.speech as speechsdk
import torch
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import webrtcvad
import socket
import warnings
import subprocess
import sys
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "YOUR_KEY_HERE")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "italynorth")

CONFIG = {
    "whisper_model": "medium",
    "translation_model": "Helsinki-NLP/opus-mt-en-ar",
    "arabic_voice": "ar-SA-HamedNeural",
    "max_segment_gap": 0.5,
    "audio_quality": "high",
    "video_codec": "libx264",
    "audio_codec": "aac",
    "max_file_size": 500 * 1024 * 1024,
    "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"],
    "temp_cleanup": True,
    "sample_rate": 22050,
    "hop_length": 512,
    "n_fft": 2048,
}

app = FastAPI(
    title="Video Dubbing API",
    description="API لدبلجة الفيديوهات من الإنجليزية إلى العربية",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

translator = None
executor = ThreadPoolExecutor(max_workers=2)

class VideoTranslatorError(Exception):
    pass

class EnhancedVideoTranslator:
    def __init__(self):
        self.temp_dir = None
        self.whisper_model = None
        self.translation_model = None
        self.tokenizer = None
        self.vad = None
        self.setup_models()

    def setup_models(self):
        try:
            self.temp_dir = tempfile.mkdtemp()
            logger.info(f"📁 Temporary directory created: {self.temp_dir}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"⚙️ Loading Whisper model on {device}...")
            self.whisper_model = whisper.load_model(CONFIG["whisper_model"], device=device)
            logger.info("✅ Whisper model loaded.")

            logger.info("⚙️ Loading translation model...")
            self.translation_model = MarianMTModel.from_pretrained(CONFIG["translation_model"])
            self.tokenizer = MarianTokenizer.from_pretrained(CONFIG["translation_model"])
            logger.info("✅ Translation model loaded.")

            try:
                self.vad = webrtcvad.Vad(2)
                logger.info("✅ VAD initialized.")
            except:
                logger.warning("⚠️ WebRTC VAD initialization failed.")
                self.vad = None

        except Exception as e:
            logger.error(f"❌ Model initialization failed: {str(e)}")
            raise VideoTranslatorError(f"Model init error: {str(e)}")

    def validate_input(self, video_path: str) -> bool:
        if not os.path.isfile(video_path):
            raise VideoTranslatorError("File does not exist.")

        size = os.path.getsize(video_path)
        if size > CONFIG["max_file_size"]:
            raise VideoTranslatorError("File too large.")

        if Path(video_path).suffix.lower() not in CONFIG["supported_formats"]:
            raise VideoTranslatorError("Unsupported video format.")

        return True

    def process_video(self, video_path: str) -> str:
        # ⚠️ Placeholder: this function should include your actual processing logic
        output_path = f"/tmp/{uuid.uuid4().hex}_dubbed.mp4"
        shutil.copy(video_path, output_path)
        logger.info(f"📽️ Processed video saved at {output_path}")
        return output_path

@app.on_event("startup")
async def startup_event():
    global translator
    try:
        translator = EnhancedVideoTranslator()
        logger.info("✅ Translator initialized successfully. Server is ready to receive requests.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize translator: {str(e)}")
        raise

@app.get("/")
async def root():
    return {
        "message": "🎬 API دبلجة الفيديوهات جاهزة للعمل!",
        "version": "1.0.0",
        "status": "✅ الموديلات محمّلة" if translator else "❌ جاري تحميل الموديلات"
    }

@app.get("/health")
async def health_check():
    status = "✅ جاهز" if translator else "❌ غير جاهز"
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "models_loaded": translator is not None,
        "message": "السيرفر يعمل والموديلات محمّلة" if translator else "الموديلات لم تُحمّل بعد"
    }

@app.post("/dub-video")
async def dub_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not translator:
        raise HTTPException(status_code=503, detail="Translator not initialized")
    
    if not file.filename.lower().endswith(tuple(CONFIG["supported_formats"])):
        raise HTTPException(status_code=400, detail="Unsupported video format.")
    
    temp_input_path = f"/tmp/{uuid.uuid4().hex}_{file.filename}"
    try:
        with open(temp_input_path, "wb") as buffer:
            content = await file.read()
            if len(content) > CONFIG["max_file_size"]:
                raise HTTPException(status_code=413, detail="File too large")
            buffer.write(content)

        loop = asyncio.get_event_loop()
        output_path = await loop.run_in_executor(executor, translator.process_video, temp_input_path)
        background_tasks.add_task(cleanup_files, temp_input_path, output_path)

        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"dubbed_{file.filename}",
            headers={"Content-Disposition": f"attachment; filename=dubbed_{file.filename}"}
        )

    except VideoTranslatorError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

def cleanup_files(*file_paths):
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"🧹 Removed temp file: {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup {file_path}: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
