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
from scipy import signal
from scipy.signal import butter, sosfilt
import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
import webrtcvad
import warnings
import subprocess
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

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
    "whisper_model": "small",
    "translation_model": "Helsinki-NLP/opus-mt-en-ar",
    "arabic_voice": "ar-SA-HamedNeural",
    "max_segment_gap": 0.5,
    "audio_quality": "high",
    "video_codec": "libx264",
    "audio_codec": "aac",
    "max_file_size": 500 * 1024 * 1024,  # 500MB
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

    def _align_audio_lengths(self, audio1: np.ndarray, audio2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """محاذاة أطوال المصفوفات الصوتية"""
        min_length = min(len(audio1), len(audio2))
        logger.debug(f"Aligning audio lengths: {len(audio1)} and {len(audio2)} to {min_length}")
        return audio1[:min_length], audio2[:min_length]

    def advanced_audio_separation(self, video_path: str) -> Tuple[str, str, str, float]:
        """فصل صوتي متقدم"""
        try:
            full_audio_path = os.path.join(self.temp_dir, "full_audio.wav")
            logger.info("Extracting full audio from video...")

            with VideoFileClip(video_path) as video:
                if video.audio is None:
                    raise VideoTranslatorError("Video has no audio track")

                duration = video.duration
                if duration <= 0:
                    raise VideoTranslatorError("Invalid video duration")

                video.audio.write_audiofile(
                    full_audio_path,
                    codec='pcm_s16le',
                    verbose=False,
                    logger=None
                )

            logger.info("Loading audio for advanced separation...")
            audio_data, sample_rate = librosa.load(full_audio_path, sr=CONFIG["sample_rate"])

            # تطبيق تقليل الضوضاء
            if CONFIG["noise_reduction_strength"] > 0:
                logger.info("Applying enhanced noise reduction...")
                audio_data = nr.reduce_noise(
                    y=audio_data,
                    sr=sample_rate,
                    prop_decrease=CONFIG["noise_reduction_strength"],
                    stationary=False,
                    n_grad_freq=2,
                    n_grad_time=4
                )

            logger.info("Performing advanced vocal isolation...")
            harmonic, percussive = librosa.effects.hpss(audio_data, margin=3.0)
            isolated_vocals = harmonic

            # محاذاة الأطوال
            original_audio, isolated_vocals = self._align_audio_lengths(audio_data, isolated_vocals)
            background_audio = original_audio - (isolated_vocals * CONFIG["vocal_isolation_strength"])

            # تطبيع الصوت
            if CONFIG["audio_normalization"]:
                isolated_vocals = librosa.util.normalize(isolated_vocals)
                background_audio = librosa.util.normalize(background_audio) * CONFIG["background_reduction_factor"]

            # حفظ ملفات الصوت
            vocals_path = os.path.join(self.temp_dir, "isolated_vocals.wav")
            background_path = os.path.join(self.temp_dir, "background_effects.wav")
            clean_vocals_path = os.path.join(self.temp_dir, "clean_vocals.wav")

            sf.write(vocals_path, isolated_vocals, sample_rate)
            sf.write(background_path, background_audio, sample_rate)
            sf.write(clean_vocals_path, isolated_vocals, sample_rate)

            logger.info(f"Advanced audio separation completed. Duration: {duration:.2f}s")
            return vocals_path, background_path, clean_vocals_path, duration

        except Exception as e:
            logger.error(f"Advanced audio separation failed: {str(e)}")
            raise VideoTranslatorError(f"Audio separation failed: {str(e)}")

    def transcribe_audio(self, audio_path: str) -> List[Dict]:
        """نسخ الصوت إلى نص"""
        try:
            logger.info("Starting audio transcription...")
            result = self.whisper_model.transcribe(
                audio_path,
                language="en",
                word_timestamps=True,
                verbose=False
            )

            segments = []
            for segment in result["segments"]:
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "confidence": getattr(segment, 'confidence', 0.9)
                })

            logger.info(f"Transcription completed. Found {len(segments)} segments")
            return segments

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise VideoTranslatorError(f"Transcription failed: {str(e)}")

    def translate_text(self, text: str) -> str:
        """ترجمة النص من الإنجليزية إلى العربية"""
        try:
            if not text.strip():
                return ""

            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = self.translation_model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
            translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)

            return translated_text.strip()

        except Exception as e:
            logger.warning(f"Translation failed for text: {text[:50]}... Error: {str(e)}")
            return text

    def synthesize_arabic_speech(self, text: str, output_path: str) -> bool:
        """تحويل النص العربي إلى كلام"""
        try:
            if not text.strip():
                return False

            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
            speech_config.speech_synthesis_voice_name = CONFIG["arabic_voice"]
            speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Wav16Khz16BitMonoPcm)

            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
            result = synthesizer.speak_text_async(text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                with open(output_path, "wb") as audio_file:
                    audio_file.write(result.audio_data)
                return True
            else:
                logger.error(f"Speech synthesis failed: {result.reason}")
                return False

        except Exception as e:
            logger.error(f"Speech synthesis failed: {str(e)}")
            return False

    def process_video(self, video_path: str) -> str:
        """معالجة الفيديو الكاملة"""
        try:
            # التحقق من صحة الملف
            self.validate_input(video_path)

            # فصل الصوت
            vocals_path, background_path, clean_vocals_path, duration = self.advanced_audio_separation(video_path)

            # نسخ الصوت
            segments = self.transcribe_audio(clean_vocals_path)

            # ترجمة وتحويل إلى كلام
            arabic_audio_segments = []
            for i, segment in enumerate(segments):
                # ترجمة النص
                arabic_text = self.translate_text(segment["text"])
                if not arabic_text:
                    continue

                # تحويل إلى كلام
                segment_audio_path = os.path.join(self.temp_dir, f"arabic_segment_{i}.wav")
                if self.synthesize_arabic_speech(arabic_text, segment_audio_path):
                    arabic_audio_segments.append({
                        "path": segment_audio_path,
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": arabic_text
                    })

            # دمج الصوت العربي
            final_arabic_audio = self._combine_arabic_audio(arabic_audio_segments, duration)

            # دمج مع الفيديو
            output_path = self._combine_with_video(video_path, final_arabic_audio, background_path)

            return output_path

        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            raise VideoTranslatorError(f"Video processing failed: {str(e)}")

    def _combine_arabic_audio(self, segments: List[Dict], total_duration: float) -> str:
        """دمج مقاطع الصوت العربية"""
        try:
            combined_audio_path = os.path.join(self.temp_dir, "combined_arabic.wav")
            
            # إنشاء صوت صامت بطول الفيديو
            silence = AudioSegment.silent(duration=int(total_duration * 1000))
            
            for segment in segments:
                if os.path.exists(segment["path"]):
                    audio_segment = AudioSegment.from_wav(segment["path"])
                    start_ms = int(segment["start"] * 1000)
                    end_ms = min(int(segment["end"] * 1000), len(silence))
                    
                    # قطع الصوت ليناسب المدة المطلوبة
                    target_duration = end_ms - start_ms
                    if len(audio_segment) > target_duration:
                        audio_segment = audio_segment[:target_duration]
                    
                    # إدراج الصوت في المكان المناسب
                    silence = silence.overlay(audio_segment, position=start_ms)
            
            silence.export(combined_audio_path, format="wav")
            return combined_audio_path

        except Exception as e:
            logger.error(f"Failed to combine Arabic audio: {str(e)}")
            raise VideoTranslatorError(f"Failed to combine Arabic audio: {str(e)}")

    def _combine_with_video(self, video_path: str, arabic_audio_path: str, background_path: str) -> str:
        """دمج الصوت العربي مع الفيديو"""
        try:
            output_path = os.path.join(self.temp_dir, f"dubbed_video_{uuid.uuid4().hex}.mp4")
            
            with VideoFileClip(video_path) as video:
                # تحميل الأصوات
                arabic_audio = AudioFileClip(arabic_audio_path)
                background_audio = AudioFileClip(background_path)
                
                # دمج الأصوات
                final_audio = CompositeAudioClip([arabic_audio, background_audio])
                
                # دمج مع الفيديو
                final_video = video.set_audio(final_audio)
                final_video.write_videofile(
                    output_path,
                    codec=CONFIG["video_codec"],
                    audio_codec=CONFIG["audio_codec"],
                    verbose=False,
                    logger=None
                )
                
                # تنظيف الذاكرة
                arabic_audio.close()
                background_audio.close()
                final_audio.close()
                final_video.close()
            
            return output_path

        except Exception as e:
            logger.error(f"Failed to combine with video: {str(e)}")
            raise VideoTranslatorError(f"Failed to combine with video: {str(e)}")

    def cleanup(self):
        """تنظيف الملفات المؤقتة"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info("Temporary files cleaned up")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp files: {str(e)}")

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

