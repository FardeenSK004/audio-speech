import os
import base64
import json
import time
from dotenv import load_dotenv

from hume.client import HumeClient
from hume.empathic_voice.types import AudioInput, SubscribeEvent, ConnectSessionSettings, ConnectSessionSettingsAudio

load_dotenv()

class HumeEVIBridge:
    def __init__(self, socket_instance, session_id, config_id=None):
        self.socket = socket_instance
        self.sid = session_id
        self.config_id = config_id or os.getenv("HUME_CONFIG_ID")
        self.api_key = os.getenv("HUME_API_KEY")
        self.client = None
        self.hume_socket = None # The actual websocket connection
        self.is_connected = False
        self.stop_event = False

    def start(self):
        """Connects to Hume EVI and starts the message processing loop (Sync)."""
        try:
            # Initialize client (Sync)
            if not self.client:
                self.client = HumeClient(api_key=self.api_key)

            print(f"[{self.sid}] Connecting to Hume EVI (Config: {self.config_id})...")
            
            # Configure audio settings (matches browser recording: 16kHz, mono, linear16)
            session_settings = ConnectSessionSettings(
                audio=ConnectSessionSettingsAudio(
                    channels=1,
                    sample_rate=16000,
                    encoding="linear16"
                )
            )
            
            # Using the chat connection context manager (Sync)
            with self.client.empathic_voice.chat.connect(
                config_id=self.config_id,
                session_settings=session_settings
            ) as hume_socket:
                self.hume_socket = hume_socket
                self.is_connected = True
                print(f"[{self.sid}] Connected to Hume EVI.")
                
                # Notify frontend we are ready
                # socket.emit is thread-safe for Flask-SocketIO (and works in greenlets)
                self.socket.emit('status', {'state': 'listening'}, room=self.sid, namespace='/hume')

                # Loop to handle incoming messages from Hume
                # The socket is a sync iterator
                for message in hume_socket:
                    if self.stop_event:
                        break
                    self._handle_hume_message(message)
                    
        except Exception as e:
            print(f"[{self.sid}] Hume Error: {e}")
            self.socket.emit('error', {'message': str(e)}, room=self.sid, namespace='/hume')
        finally:
            self.is_connected = False
            self.hume_socket = None
            print(f"[{self.sid}] Hume connection closed.")

    def stop(self):
        self.stop_event = True
        self.is_connected = False # Immediately flag as disconnected to stop sending audio

    def send_audio(self, audio_data: bytes):
        """Sends raw PCM audio (bytes) to Hume (Sync)."""
        if self.is_connected and self.hume_socket:
            try:
                # Convert bytes to base64 string
                b64_data = base64.b64encode(audio_data).decode('utf-8')
                
                # Create AudioInput message
                audio_input = AudioInput(data=b64_data)
                
                # Send via socket (Sync)
                self.hume_socket.send_audio_input(audio_input)
                
            except Exception as e:
                print(f"[{self.sid}] Error sending audio: {e}")

    def _handle_hume_message(self, message: SubscribeEvent):
        """Dispatches Hume events to the browser."""
        # message is a SubscribeEvent object
        
        msg_type = message.type
        
        if msg_type == "user_message":
            # Transcription of what the user said
            text = message.message.content
            
            self.socket.emit('transcription', {
                'text': text,
                'is_final': True
            }, room=self.sid, namespace='/hume')
            
            # Extract emotions if available
            if message.models and message.models.prosody and message.models.prosody.scores:
                scores = message.models.prosody.scores
                # scores is a Pydantic model, need to convert to dict
                scores_dict = scores.dict()
                top_3 = dict(sorted(scores_dict.items(), key=lambda item: item[1] or 0, reverse=True)[:3])
                self.socket.emit('emotions', {'scores': top_3}, room=self.sid, namespace='/hume')
            
        elif msg_type == "assistant_message":
            # The text response from EVI
            text = message.message.content
            self.socket.emit('llm_response', {'text': text}, room=self.sid, namespace='/hume')
            
            # Assistant emotions (optional)
            if message.models and message.models.prosody and message.models.prosody.scores:
                scores = message.models.prosody.scores
                # scores is a Pydantic model, need to convert to dict
                scores_dict = scores.dict()
                top_3 = dict(sorted(scores_dict.items(), key=lambda item: item[1] or 0, reverse=True)[:3])
                self.socket.emit('emotions', {'scores': top_3, 'source': 'assistant'}, room=self.sid, namespace='/hume')

        elif msg_type == "audio_output":
            # Binary audio data
            # message.data is a base64 string
            try:
                audio_bytes = base64.b64decode(message.data)
                self.socket.emit('bot_audio', {'audio': audio_bytes}, room=self.sid, namespace='/hume')
            except Exception as e:
                print(f"[{self.sid}] Audio decode error: {e}")
            
        elif msg_type == "error":
            print(f"[{self.sid}] Hume Error Event: {message.code} - {message.message}")
            self.socket.emit('error', {'message': message.message}, room=self.sid, namespace='/hume')