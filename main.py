import os
import sounddevice as sd
import numpy as np
import soundfile as sf
import openai
from google.cloud import texttospeech
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.playback import play

# Set API Credentials
openai.api_type = "azure"
openai.api_base = ""
openai.api_version = "2024-02-15-preview"
openai.api_key = ""
MODEL_NAME = ""

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""


#Step 1: Getting the list of AudibleList
def get_valid_input_device():
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            return i
    return None


#Step 2: Start Recording with duration/sample rate
def record_audio(filename="input.wav", duration=5):
    device_id = get_valid_input_device()
    if device_id is None:
        print("❌ No valid microphone found!")
        return False

    print(f"🎤 Using input device {device_id}... Recording...")
    try:
        sample_rate = int(sd.query_devices(device_id)["default_samplerate"])
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16,
            device=device_id)
        sd.wait()
        sf.write(filename, recording, sample_rate)
        print("✅ Recording saved as", filename)
        return True
    except Exception as e:
        print("❌ Error during recording:", e)
        return False

#Step 3: Transforming audio to text format.
def transcribe_audio(audio_file="input.wav"):
    if not os.path.exists(audio_file):
        print("❌ Audio file not found!")
        return ""

    print("📝 Transcribing audio...")
    model = WhisperModel("base", device="cpu", compute_type="float32")
    segments, _ = model.transcribe(audio_file)
    transcript = " ".join(segment.text for segment in segments).strip()
    print("✅ Transcription:", transcript)
    return transcript

#Step 4: Providing the transcribed text to azure openai LLM.
def generate_response(user_input):
    try:
        response = openai.ChatCompletion.create(
            engine=MODEL_NAME,
            messages=[{"role": "user", "content": user_input}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ OpenAI API error:", e)
        return "Error: Unable to get response."

#Step 5: Again using TTS client by google cloud to generate output generate file
def text_to_speech(text, output_file="output.mp3"):
    print("🔊 Converting text to speech...")
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        with open(output_file, "wb") as out:
            out.write(response.audio_content)
        print("✅ Response saved as", output_file)

        audio = AudioSegment.from_mp3(output_file)
        play(audio)
    except Exception as e:
        print("❌ TTS Error:", e)

#Step 6: While loop until exit command is provided by user.
def voice_chatbot():
    print("🎙️ Voice chatbot activated. Say 'exit' to stop.")
    while True:
        if not record_audio():
            break

        text = transcribe_audio()
        if not text:
            print("❌ Failed to transcribe audio.")
            continue

        print("🗣️ You said:", text)

        if "exit" in text.lower():
            print("👋 Exiting voice chatbot...")
            break

        response = generate_response(text)
        print("🤖 Chatbot:", response)

        text_to_speech(response)

if __name__ == "__main__":
    voice_chatbot()
