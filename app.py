import eventlet
eventlet.monkey_patch()

import os
import io
import time
import queue
import threading
import numpy as np
import webrtcvad
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from openai import OpenAI
from dotenv import load_dotenv
from stt import STT
from tts import TTS

print("--- Starting Server Initialization ---")
load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Initialize modules
print("Initializing STT model (faster-whisper)...")
stt_model = STT(model_size="base")
print("STT model ready.")

print("Initializing TTS engine...")
tts_engine = TTS()
print("TTS engine ready.")

print("Initializing OpenAI client...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("OpenAI client ready.")

# ---- Config ----
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
VAD_MODE = 0 # Most sensitive mode
SYSTEM_PROMPT = "You are a helpful, extremely concise live assistant. Respond naturally but keep answers under 20 words where possible."

# Session state storage
sessions = {}

class SessionState:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.vad_buffer = bytearray()
        self.audio_buffer = []
        self.is_recording = False
        self.silent_frames = 0
        self.speech_frames = 0
        self.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.last_activity = time.time()
        self.processing = False

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    print(f"Client connected: {sid}")
    sessions[sid] = SessionState()

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    print(f"Client disconnected: {sid}")
    if sid in sessions:
        del sessions[sid]

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    sid = request.sid
    if sid not in sessions:
        sessions[sid] = SessionState()
    
    state = sessions[sid]
    state.last_activity = time.time()
    
    if not hasattr(state, 'chunk_count'): state.chunk_count = 0
    state.chunk_count += 1
    
    # Calculate RMS energy to check for silence
    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float64)
    rms = np.sqrt(np.mean(audio_np**2))
    
    if not hasattr(state, 'rms_buffer'): state.rms_buffer = []
    state.rms_buffer.append(rms)

    if state.chunk_count % 50 == 0:
        avg_rms = sum(state.rms_buffer) / len(state.rms_buffer) if state.rms_buffer else 0
        state.rms_buffer = [] # Reset buffer
        print(f"[{sid}] Received 50 audio chunks (Total: {state.chunk_count}) | Avg RMS: {avg_rms:.2f}")

    if state.processing:
        return

    
    # Add to internal VAD buffer
    state.vad_buffer.extend(data)
    
    # Process all complete 30ms frames (960 bytes) in the buffer
    FRAME_BYTES = 960 # 480 samples * 2 bytes
    
    while len(state.vad_buffer) >= FRAME_BYTES:
        frame = bytes(state.vad_buffer[:FRAME_BYTES])
        del state.vad_buffer[:FRAME_BYTES]
        
        is_speech = False
        try:
            is_speech = state.vad.is_speech(frame, SAMPLE_RATE)
        except Exception as e:
            print(f"VAD Error: {e}")
            continue

        if is_speech:
            if not state.is_recording:
                state.is_recording = True
                print(f"[{sid}] VAD: Speech detected. Starting recording.")
                emit('status', {'state': 'listening'})
            
            state.audio_buffer.append(frame)
            state.speech_frames += 1
            state.silent_frames = 0
        else:
            if state.is_recording:
                state.silent_frames += 1
                state.audio_buffer.append(frame)
                
                # End detection: 800ms of silence
                if state.silent_frames > (800 / FRAME_DURATION):
                    state.is_recording = False
                    emit('status', {'state': 'processing'})
                    
                    audio_to_process = b"".join(state.audio_buffer)
                    state.audio_buffer = []
                    state.silent_frames = 0
                    state.speech_frames = 0
                    
                    socketio.start_background_task(process_speech, sid, audio_to_process)

def process_speech(sid, audio_bytes):
    if sid not in sessions:
        return
        
    state = sessions[sid]
    state.processing = True
    
    try:
        # Step 1: Transcribe
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        text, info = stt_model.transcribe(audio_np)
        
        if not text.strip() or len(text.strip()) < 2:
            state.processing = False
            socketio.emit('status', {'state': 'ready'}, room=sid)
            return

        print(f"[{sid}] You: {text}")
        socketio.emit('transcription', {'text': text}, room=sid)
        state.conversation.append({"role": "user", "content": text})

        # Step 2 & 3: Streaming LLM and TTS
        print(f"[{sid}] Starting LLM stream...")
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=state.conversation,
            temperature=0.7,
            stream=True
        )

        full_reply = ""
        sentence_buffer = ""
        
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                full_reply += token
                sentence_buffer += token
                # Emit token immediately for UI
                socketio.emit('llm_token', {'token': token}, room=sid)

                # Check for sentence boundaries
                if any(punct in sentence_buffer for punct in ['.', '!', '?', '\n']) and len(sentence_buffer) > 20:
                    sentence = sentence_buffer.strip()
                    sentence_buffer = ""
                    # Trigger TTS in background for this sentence
                    socketio.start_background_task(generate_and_emit_tts, sid, sentence)

        # Final sentence if any remains
        if sentence_buffer.strip():
            socketio.start_background_task(generate_and_emit_tts, sid, sentence_buffer.strip())

        state.conversation.append({"role": "assistant", "content": full_reply})
        print(f"[{sid}] Bot (Full): {full_reply}")

    except Exception as e:
        print(f"Processing error: {e}")
        try:
            socketio.emit('error', {'message': str(e)}, room=sid)
        except:
            pass
    
    state.processing = False
    socketio.emit('status', {'state': 'ready'}, room=sid)

def generate_and_emit_tts(sid, text):
    try:
        audio_response_bytes = tts_engine.get_audio_bytes(text)
        if audio_response_bytes:
            socketio.emit('bot_audio', audio_response_bytes, room=sid)
    except Exception as e:
        print(f"TTS Streaming error: {e}")

if __name__ == '__main__':
    print("--- Server Starting on http://0.0.0.0:6123 ---")
    socketio.run(app, host='0.0.0.0', port=6123)
