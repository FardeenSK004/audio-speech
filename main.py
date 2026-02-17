import os
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import webrtcvad
from openai import OpenAI
from dotenv import load_dotenv
from stt import STT
from tts import TTS
from colorama import Fore, Back, Style, init

init(autoreset=True)

# ---- Config ----
load_dotenv()
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
VAD_MODE = 3
CHANNELS = 1

# Energy threshold (0.0 to 1.0) - Adjust based on environment
# If you are hearing yourself too much, increase this or check diagnostics.
ENERGY_THRESHOLD = 0.02 # Increased from 0.01

# System Prompt
SYSTEM_PROMPT = "You are a helpful, extremely concise live assistant. Respond naturally but keep answers under 20 words where possible."

class LiveAssistant:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.stt = STT(model_size="base")
        self.tts = TTS()
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.audio_queue = queue.Queue()
        self.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.is_recording = False
        
        # Echo suppression flags
        self.last_stop_talking_time = 0
        self.playback_start_time = 0
        self.interruption_cooldown = 2.0  # Increased to 2 seconds
        self.post_speech_silence = 0.8    # Increased to 0.8 seconds to catch echo tail

    def audio_callback(self, indata, frames, time, status):
        self.audio_queue.put(indata.copy())

    def run(self):
        print(f"{Fore.GREEN}{Style.BRIGHT}Live Assistant Started. Speak to begin...{Style.RESET_ALL}")
        
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32',
            blocksize=FRAME_SIZE,
            callback=self.audio_callback
        ):
            audio_buffer = []
            silent_frames = 0
            speech_frames_count = 0
            was_playing = False
            
            while True:
                try:
                    frame = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                now = time.time()
                is_currently_playing = self.tts.is_playing
                
                # Detect transition from playing to stopped (natural finish)
                if was_playing and not is_currently_playing:
                    self.last_stop_talking_time = now
                    # Flush queue to avoid trailing bot voice
                    while not self.audio_queue.empty(): self.audio_queue.get()
                
                was_playing = is_currently_playing
                
                # Energy calculation (RMS)
                rms = np.sqrt(np.mean(frame**2))
                
                # Diagnostic help: Uncomment to see energy levels and calibrate ENERGY_THRESHOLD
                # if rms > 0.005: print(f" Energy: {rms:.4f}") 
                
                # Convert to int16 for VAD
                audio_int16 = (frame[:, 0] * 32768).astype(np.int16)
                is_speech = self.vad.is_speech(audio_int16.tobytes(), SAMPLE_RATE)
                
                # Apply energy threshold to filter out low-level noise
                if rms < ENERGY_THRESHOLD:
                    is_speech = False

                # --- ECHO SUPPRESSION & INTERRUPTION LOGIC ---
                if is_currently_playing:
                    if self.playback_start_time == 0:
                        self.playback_start_time = now
                    
                    if now - self.playback_start_time < self.interruption_cooldown:
                        continue
                    
                    if is_speech:
                        print(f"\n{Fore.YELLOW}[Interrupted!]{Style.RESET_ALL}")
                        self.tts.stop()
                        self.last_stop_talking_time = now
                        self.playback_start_time = 0
                        audio_buffer = []
                        while not self.audio_queue.empty(): self.audio_queue.get()
                        continue
                else:
                    self.playback_start_time = 0
                    if now - self.last_stop_talking_time < self.post_speech_silence:
                        continue

                # --- RECORDING LOGIC ---
                if is_speech:
                    speech_frames_count += 1
                    # Require at least 3 frames (~90ms) of consistent speech to start recording
                    if not self.is_recording and speech_frames_count > 3:
                        print(f"{Fore.MAGENTA}Listening...{Style.RESET_ALL}", end="", flush=True)
                        self.is_recording = True
                    
                    if self.is_recording:
                        audio_buffer.append(frame)
                    silent_frames = 0
                else:
                    if self.is_recording:
                        silent_frames += 1
                        audio_buffer.append(frame)
                        
                        # Use a dynamic silence window (800ms)
                        if silent_frames > (800 / FRAME_DURATION):
                            print(f"{Fore.GREEN} Done.{Style.RESET_ALL}")
                            # Final sanity check: was it long enough?
                            if len(audio_buffer) > (SAMPLE_RATE * 0.5 / FRAME_SIZE):
                                threading.Thread(target=self.process_audio, args=(audio_buffer.copy(),), daemon=True).start()
                            
                            audio_buffer = []
                            self.is_recording = False
                            silent_frames = 0
                            speech_frames_count = 0
                    else:
                        speech_frames_count = 0

    def process_audio(self, audio_buffer):
        audio_data = np.concatenate(audio_buffer, axis=0).flatten()
        
        print(f"{Fore.CYAN}Transcribing...{Style.RESET_ALL}")
        text, info = self.stt.transcribe(audio_data)
        
        # Filter out junk transcriptions
        if not text.strip() or len(text.strip()) < 3:
            return

        print(f"{Fore.BLUE}{Style.BRIGHT}You: {text}{Style.RESET_ALL}")
        self.conversation.append({"role": "user", "content": text})

        print(f"{Fore.YELLOW}Thinking...{Style.RESET_ALL}")
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.conversation,
                temperature=0.7
            )
            reply = response.choices[0].message.content
            print(f"{Fore.GREEN}{Style.BRIGHT}Bot: {reply}{Style.RESET_ALL}")
            self.conversation.append({"role": "assistant", "content": reply})
            self.tts.speak(reply)
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    assistant = LiveAssistant()
    try:
        assistant.run()
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Exiting...{Style.RESET_ALL}")
