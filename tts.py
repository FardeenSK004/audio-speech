from openai import OpenAI
import os
import io
import sounddevice as sd
import soundfile as sf
import threading
from dotenv import load_dotenv
from colorama import Fore, Style, init

load_dotenv()
init(autoreset=True)

class TTS:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "tts-1"
        self.is_playing = False

    def get_audio_bytes(self, text):
        """Synthesizes text and returns the raw audio bytes (MP3)."""
        if not text:
            return None
            
        try:
            print(f"{Fore.CYAN}Generating speech...{Style.RESET_ALL}")
            response = self.client.audio.speech.create(
                model=self.model,
                voice="nova",
                input=text,
            )
            return response.content
        except Exception as e:
            print(f"{Fore.RED}TTS Error: {e}{Style.RESET_ALL}")
            return None

    def _play_audio_local(self, data, samplerate):
        """Internal helper for local testing."""
        self.is_playing = True
        try:
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"{Fore.RED}Playback error: {e}{Style.RESET_ALL}")
        finally:
            self.is_playing = False

    def speak_local(self, text):
        """Synthesizes text and plays it directly (for local debugging)."""
        audio_data = self.get_audio_bytes(text)
        if audio_data:
            with io.BytesIO(audio_data) as audio_file:
                data, samplerate = sf.read(audio_file)
                threading.Thread(target=self._play_audio_local, args=(data, samplerate), daemon=True).start()

    def stop_local(self):
        sd.stop()
        self.is_playing = False

if __name__ == "__main__":
    tts = TTS()
    tts.speak_local("This is a local test of the refactored TTS class.")
    import time
    time.sleep(3)