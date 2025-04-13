import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import speech_recognition as sr
from gtts import gTTS
import pygame
import wave
import pyaudio
import numpy as np
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceModel:
    """
    Model to handle voice interactions including speech-to-text and text-to-speech.
    """
    
    def __init__(self):
        """
        Initialize the voice model with speech recognition and text-to-speech capabilities.
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Audio settings
        self.chunk = 1024
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 44100
        
        # Initialize Pygame for audio playback
        pygame.mixer.init()
        
        # Create temp directory for audio files
        self.temp_dir = Path(tempfile.gettempdir()) / "farm_chatbot_audio"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported languages for speech recognition
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'hi': 'Hindi',
            'bn': 'Bengali',
            'ar': 'Arabic'
        }
        
        logger.info("Voice model initialized")
        
    def record_audio(self, duration: float = 5.0) -> Optional[str]:
        """
        Record audio from microphone.
        
        Args:
            duration (float): Recording duration in seconds
            
        Returns:
            str: Path to the recorded audio file
        """
        try:
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            
            # Open stream
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            # Create a temporary WAV file
            temp_file = self.temp_dir / f"recording_{os.getpid()}.wav"
            
            # Record audio
            frames = []
            for _ in range(0, int(self.rate / self.chunk * duration)):
                data = stream.read(self.chunk)
                frames.append(data)
            
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save the recorded data as a WAV file
            with wave.open(str(temp_file), 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
            
            return str(temp_file)
            
        except Exception as e:
            logger.error(f"Error recording audio: {str(e)}")
            return None
            
    def speech_to_text(self, audio_file: str, language: str = 'en') -> Tuple[bool, str]:
        """
        Convert speech to text using Google Speech Recognition.
        
        Args:
            audio_file (str): Path to the audio file
            language (str): Language code for speech recognition
            
        Returns:
            tuple: (success, text or error message)
        """
        try:
            # Check if language is supported
            if language not in self.supported_languages:
                return False, f"Unsupported language: {language}"
            
            # Load the audio file
            with sr.AudioFile(audio_file) as source:
                # Record audio to memory
                audio = self.recognizer.record(source)
                
                # Recognize speech using Google Speech Recognition
                text = self.recognizer.recognize_google(
                    audio,
                    language=language
                )
                
                return True, text
                
        except sr.UnknownValueError:
            return False, "Could not understand audio"
        except sr.RequestError as e:
            return False, f"Could not request results: {str(e)}"
        except Exception as e:
            logger.error(f"Error in speech to text: {str(e)}")
            return False, f"Error processing audio: {str(e)}"
            
    def text_to_speech(self, text: str, language: str = 'en', speed: float = 1.0) -> Optional[str]:
        """
        Convert text to speech using Google Text-to-Speech.
        
        Args:
            text (str): Text to convert to speech
            language (str): Language code for speech synthesis
            speed (float): Speech speed multiplier
            
        Returns:
            str: Path to the generated audio file
        """
        try:
            # Check if language is supported
            if language not in self.supported_languages:
                logger.error(f"Unsupported language: {language}")
                return None
            
            # Create a temporary file for the audio
            temp_file = self.temp_dir / f"tts_{os.getpid()}.mp3"
            
            # Generate speech
            tts = gTTS(
                text=text,
                lang=language,
                slow=(speed < 1.0)
            )
            
            # Save the audio file
            tts.save(str(temp_file))
            
            return str(temp_file)
            
        except Exception as e:
            logger.error(f"Error in text to speech: {str(e)}")
            return None
            
    def play_audio(self, audio_file: str) -> bool:
        """
        Play an audio file.
        
        Args:
            audio_file (str): Path to the audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load and play the audio file
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            return True
            
        except Exception as e:
            logger.error(f"Error playing audio: {str(e)}")
            return False
            
    def cleanup_audio_file(self, audio_file: str) -> bool:
        """
        Clean up an audio file.
        
        Args:
            audio_file (str): Path to the audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if os.path.exists(audio_file):
                os.remove(audio_file)
            return True
        except Exception as e:
            logger.error(f"Error cleaning up audio file: {str(e)}")
            return False
            
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get list of supported languages for voice interactions.
        
        Returns:
            dict: Dictionary of language codes and names
        """
        return self.supported_languages
        
    def is_language_supported(self, lang_code: str) -> bool:
        """
        Check if a language is supported for voice interactions.
        
        Args:
            lang_code (str): Language code to check
            
        Returns:
            bool: True if language is supported
        """
        return lang_code in self.supported_languages 