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
stt_model = STT(model_size="tiny")
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
SYSTEM_PROMPT = "Be as pookie as possible and respond to me in a cute way without any emojis"

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

@app.route('/ring-ui')
def ring_ui():
    return render_template('ring_ui.html')

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
                
                # End detection: 300ms of silence (reduced for lower latency)
                if state.silent_frames > (300 / FRAME_DURATION):
                    state.is_recording = False
                    emit('status', {'state': 'processing'})
                    
                    audio_to_process = b"".join(state.audio_buffer)
                    state.audio_buffer = []
                    state.silent_frames = 0
                    state.speech_frames = 0
                    
                    socketio.start_background_task(process_speech, sid, audio_to_process)

def tts_worker(sid, tts_queue):
    """Background worker that processes TTS requests from the queue."""
    while True:
        item = tts_queue.get()
        if item is None:  # Sentinel to stop
            break
        text, index = item
        try:
            t_start = time.time()
            print(f"[{sid}] TTS chunk {index}: {text[:40]}...")
            audio_response_bytes = tts_engine.get_audio_bytes(text)
            if audio_response_bytes:
                elapsed = time.time() - t_start
                print(f"[{sid}] TTS chunk {index} ready ({len(audio_response_bytes)} bytes, {elapsed:.2f}s)")
                socketio.emit('bot_audio', {'audio': audio_response_bytes, 'index': index}, room=sid)
            else:
                print(f"[{sid}] TTS generation failed for chunk {index}, sending skip")
                socketio.emit('bot_audio_skip', {'index': index}, room=sid)
        except Exception as e:
            print(f"TTS worker error (chunk {index}): {e}")

def process_speech(sid, audio_bytes):
    if sid not in sessions:
        return
        
    state = sessions[sid]
    state.processing = True
    
    try:
        # Step 1: Transcribe
        t0 = time.time()
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        text, info = stt_model.transcribe(audio_np)
        stt_time = time.time() - t0
        
        if not text.strip() or len(text.strip()) < 2:
            state.processing = False
            socketio.emit('status', {'state': 'ready'}, room=sid)
            return

        print(f"[{sid}] You: {text} (STT: {stt_time:.2f}s)")
        socketio.emit('transcription', {'text': text}, room=sid)
        state.conversation.append({"role": "user", "content": text})

        # Step 2 & 3: Streaming LLM with PARALLEL TTS
        print(f"[{sid}] Starting LLM stream...")
        t1 = time.time()
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=state.conversation,
            temperature=0.7,
            stream=True
        )

        # Start TTS worker thread — it processes sentences as they arrive
        tts_queue = queue.Queue()
        tts_thread = threading.Thread(target=tts_worker, args=(sid, tts_queue), daemon=True)
        tts_thread.start()

        full_reply = ""
        sentence_buffer = ""
        chunk_index = 0
        first_token_time = None
        
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                if first_token_time is None:
                    first_token_time = time.time() - t1
                    print(f"[{sid}] First LLM token in {first_token_time:.2f}s")
                
                full_reply += token
                sentence_buffer += token
                # Emit token immediately for UI
                socketio.emit('llm_token', {'token': token}, room=sid)

                # Check for sentence boundaries (only proper sentence endings)
                if any(p in sentence_buffer for p in ['.', '!', '?', '\n']) and len(sentence_buffer) > 15:
                    sentence = sentence_buffer.strip()
                    sentence_buffer = ""
                    # Non-blocking: push to TTS worker queue
                    tts_queue.put((sentence, chunk_index))
                    chunk_index += 1

        # Final sentence if any remains
        if sentence_buffer.strip():
            tts_queue.put((sentence_buffer.strip(), chunk_index))

        # Signal TTS worker to finish and wait
        tts_queue.put(None)
        tts_thread.join()

        # Tell frontend how many chunks to expect
        socketio.emit('tts_complete', {'total_chunks': chunk_index}, room=sid)

        total_time = time.time() - t0
        state.conversation.append({"role": "assistant", "content": full_reply})
        print(f"[{sid}] Bot (Full): {full_reply}")
        print(f"[{sid}] Total pipeline: {total_time:.2f}s")

    except Exception as e:
        print(f"Processing error: {e}")
        try:
            socketio.emit('error', {'message': str(e)}, room=sid)
        except:
            pass
    
    state.processing = False
    socketio.emit('status', {'state': 'ready'}, room=sid)

# ---- Hume EVI Integration ----
from huss import HumeEVIBridge

sessions_hume = {}

@app.route('/hume')
def hume_ui():
    return render_template('hume.html')

@socketio.on('connect', namespace='/hume')
def handle_hume_connect():
    sid = request.sid
    print(f"Hume Client connected: {sid}")
    bridge = HumeEVIBridge(socketio, sid)
    sessions_hume[sid] = bridge
    
    # Start the bridge in a background task (greenlet)
    # Since huss.py is now sync (using websockets.sync), this works fine with eventlet.
    socketio.start_background_task(bridge.start)

@socketio.on('disconnect', namespace='/hume')
def handle_hume_disconnect():
    sid = request.sid
    print(f"Hume Client disconnected: {sid}")
    if sid in sessions_hume:
        sessions_hume[sid].stop()
        del sessions_hume[sid]

@socketio.on('audio_chunk', namespace='/hume')
def handle_hume_audio(data):
    sid = request.sid
    if sid in sessions_hume:
        # Use simple sync method
        sessions_hume[sid].send_audio(data)

if __name__ == '__main__':
    print("--- Server Starting on http://0.0.0.0:6123 ---")
    socketio.run(app, host='0.0.0.0', port=6123)
