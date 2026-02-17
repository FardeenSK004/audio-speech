from faster_whisper import WhisperModel
import numpy as np

class STT:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_data: np.ndarray):
        """
        Transcribe audio data (numpy array).
        """
        # faster-whisper can take a numpy array directly
        segments, info = self.model.transcribe(audio_data, beam_size=1)
        text = "".join([segment.text for segment in segments]).strip()
        return text, info

# For testing
if __name__ == "__main__":
    stt = STT()
    # Dummy transcription attempt if file exists
    import os
    if os.path.exists("output.wav"):
        import soundfile as sf
        audio, _ = sf.read("output.wav")
        text, info = stt.transcribe(audio)
        print(f"Transcription: {text}")
