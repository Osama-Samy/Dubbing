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

# تجاهل التحذيرات غير المهمة
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Azure Speech Service credentials
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "FMJPLiTea92XmK7ZNqv3CscieRTdQNU5ihZ26RFUrHACpxPKTLiMJQQJ99BFACgEuAYXJ3w3AAAYACOGCMRL")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "italynorth")

# Enhanced Configuration
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
    "vocal_isolation_strength": 0.9,
    "noise_reduction_strength": 0.4,
    "voice_enhancement": True,
    "preserve_effects": True,
    "audio_normalization": True,
    "dynamic_range_compression": 0.8,
    "sample_rate": 22050,
    "hop_length": 512,
    "n_fft": 2048,
    "preserve_audio_length": True,
    "stft_center": True,
    "audio_padding_mode": "reflect",
    "length_tolerance": 1024,
    "translation_confidence_threshold": -0.8,
    "audio_sync_tolerance": 0.1,
    "voice_clone_quality": "high",
    "background_reduction_factor": 0.15
}

# Initialize FastAPI app
app = FastAPI(
    title="Video Dubbing API",
    description="API لدبلجة الفيديوهات من الإنجليزية إلى العربية",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
translator = None
executor = ThreadPoolExecutor(max_workers=2)

class VideoTranslatorError(Exception):
    """Custom exception for video translator errors"""
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
        """Initialize all models with error handling"""
        try:
            self.temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory: {self.temp_dir}")

            # Load Whisper model
            logger.info("Loading Whisper model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = whisper.load_model(CONFIG["whisper_model"], device=device)
            logger.info(f"Whisper model loaded on {device}")

            # Load translation model
            logger.info("Loading translation model...")
            self.translation_model = MarianMTModel.from_pretrained(CONFIG["translation_model"])
            self.tokenizer = MarianTokenizer.from_pretrained(CONFIG["translation_model"])
            logger.info("Translation model loaded successfully")

            # Initialize Voice Activity Detection
            try:
                self.vad = webrtcvad.Vad(2)
                logger.info("Voice Activity Detection initialized")
            except:
                logger.warning("Could not initialize WebRTC VAD, using fallback method")
                self.vad = None

        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise VideoTranslatorError(f"Model initialization failed: {str(e)}")

    def validate_input(self, video_path: str) -> bool:
        """Validate input video file"""
        if not video_path:
            raise VideoTranslatorError("No video file provided")

        if not os.path.isfile(video_path):
            raise VideoTranslatorError("Video file does not exist")

        file_size = os.path.getsize(video_path)
        if file_size > CONFIG["max_file_size"]:
            raise VideoTranslatorError(f"File too large. Maximum size: {CONFIG['max_file_size']/1024/1024:.1f}MB")

        file_ext = Path(video_path).suffix.lower()
        if file_ext not in CONFIG["supported_formats"]:
            raise VideoTranslatorError(f"Unsupported format. Supported: {', '.join(CONFIG['supported_formats'])}")

        return True

    # (The rest of the methods such as audio separation, transcription, etc. will be the same as in your original code)

# Initialize translator on startup
@app.on_event("startup")
async def startup_event():
    global translator
    try:
        translator = EnhancedVideoTranslator()
        logger.info("Video translator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize translator: {str(e)}")
        raise

@app.get("/")
async def root():
    """نقطة البداية للـ API"""
    return {
        "message": "مرحباً بك في API دبلجة الفيديوهات",
        "version": "1.0.0",
        "description": "API لدبلجة الفيديوهات من الإنجليزية إلى العربية"
    }

@app.get("/health")
async def health_check():
    """فحص صحة الخدمة"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": translator is not None
    }

@app.post("/dub-video")
async def dub_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """دبلجة فيديو من الإنجليزية إلى العربية"""
    if not translator:
        raise HTTPException(status_code=503, detail="Translator not initialized")
    
    # التحقق من نوع الملف
    if not file.filename.lower().endswith(tuple(CONFIG["supported_formats"])):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported: {', '.join(CONFIG['supported_formats'])}"
        )
    
    # حفظ الملف المرفوع
    temp_input_path = f"/tmp/{uuid.uuid4().hex}_{file.filename}"
    try:
        with open(temp_input_path, "wb") as buffer:
            content = await file.read()
            if len(content) > CONFIG["max_file_size"]:
                raise HTTPException(status_code=413, detail="File too large")
            buffer.write(content)
        
        # معالجة الفيديو في خيط منفصل
        loop = asyncio.get_event_loop()
        output_path = await loop.run_in_executor(
            executor, 
            translator.process_video, 
            temp_input_path
        )
        
        # إضافة مهمة تنظيف في الخلفية
        background_tasks.add_task(cleanup_files, temp_input_path, output_path)
        
        # إرجاع الملف المدبلج
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"dubbed_{file.filename}",
            headers={"Content-Disposition": f"attachment; filename=dubbed_{file.filename}"}
        )
        
    except VideoTranslatorError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

def cleanup_files(*file_paths):
    """تنظيف الملفات"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
