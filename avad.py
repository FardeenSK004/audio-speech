import webrtcvad
import sounddevice as sd
import numpy as np

# ---- Config ----
SAMPLE_RATE = 16000        # must be 8000, 16000, 32000, or 48000
FRAME_DURATION = 30        # 10, 20, or 30 ms only
CHANNELS = 1
VAD_MODE = 2               # 0-3 (3 = most aggressive)

# Frame size in samples
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

vad = webrtcvad.Vad(VAD_MODE)

print("Speak into the microphone...")

def callback(indata, frames, time, status):
    if status:
        print(status)

    # Convert float32 -> int16 (required by webrtcvad)
    audio = (indata[:, 0] * 32768).astype(np.int16)
    audio_bytes = audio.tobytes()

    if len(audio) == FRAME_SIZE:
        is_speech = vad.is_speech(audio_bytes, SAMPLE_RATE)
        print("Speech" if is_speech else "Silence")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype='float32',
    blocksize=FRAME_SIZE,
    callback=callback,
):
    sd.sleep(20000)  # run for 20 seconds
