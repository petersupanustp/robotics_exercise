"""Voice-Chat-Demo: Spracheingabe -> Whisper -> lokales LLM -> Kokoro-TTS -> Reachy Mini.

Ablauf pro Runde:
    1. Nutzer drückt ENTER, es wird ein paar Sekunden lang aufgenommen.
    2. Die Aufnahme wird lokal mit Whisper transkribiert.
    3. Der erkannte Text wird an ein lokales LLM (Ollama, OpenAI-kompatible API) geschickt.
    4. Die Antwort des LLM wird mit Kokoro-82M (kokoro-onnx) in Audio umgewandelt.
    5. Das erzeugte Audio wird über die reachy_mini-API auf dem Roboter abgespielt.

Hinweis: Kokoro-82M unterstützt offiziell kein Deutsch, daher läuft diese Demo komplett
auf Englisch (Spracheingabe, LLM-Antwort und Sprachausgabe).

Voraussetzung: Die Kokoro-ONNX-Modelldateien müssen im Unterordner `tts_models/` liegen:
    - tts_models/kokoro-v1.0.onnx
    - tts_models/voices-v1.0.bin
Download (model-files-v1.0, siehe decisions.md zur Versionswahl):
    https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0
"""

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pyexpat  # noqa: F401  # must be loaded before reachy_mini (GStreamer ships a conflicting libexpat.dll)
import soundfile as sf
import sounddevice as sd
import whisper
from kokoro_onnx import Kokoro
from openai import OpenAI
from reachy_mini import ReachyMini

MODEL_DIR = Path(__file__).parent / "tts_models"
KOKORO_MODEL_PATH = MODEL_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = MODEL_DIR / "voices-v1.0.bin"
KOKORO_VOICE = "af_heart"

# 1. Lokales Whisper-Modell laden (Spracherkennung, Englisch)
print("Lade lokales Whisper-Modell ('base')...")
whisper_model = whisper.load_model("base")

# 2. Verbindung zur lokalen Ollama-Instanz via OpenAI-API herstellen
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ein API-Key wird von Ollama nicht benötigt, darf aber nicht leer sein
)

# 3. Lokales Kokoro-82M-TTS-Modell laden (Sprachsynthese, Englisch)
print("Lade lokales Kokoro-TTS-Modell...")
tts = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))


def record_and_transcribe(duration: float = 5.0, sample_rate: int = 16000) -> str:
    """Nimmt für `duration` Sekunden Audio auf und gibt den transkribierten Text zurück."""
    print(f"\n[Mikro] Aufnahme startet für {duration} Sekunden... Bitte sprechen (Englisch)!")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()  # Warten, bis die Aufnahmezeit abgelaufen ist
    print("[Mikro] Aufnahme beendet. Transkribiere lokal...")

    # Whisper akzeptiert direkt ein float32-Mono-Array mit 16 kHz, kein ffmpeg/WAV nötig.
    audio = np.asarray(audio_data, dtype=np.float32).flatten()
    result = whisper_model.transcribe(audio, language="en")
    return result["text"].strip()


def ask_llm(user_text: str) -> str:
    """Schickt den erkannten Text an das lokale LLM und gibt eine kurze, sprechbare Antwort zurück."""
    system_prompt = (
        "You are the voice and brain of the Reachy Mini robot. "
        "Reply in 1-3 short, natural spoken sentences in English. "
        "Do not use markdown, bullet points, emojis or code."
    )
    try:
        response = client.chat.completions.create(
            model="phi4-mini",  # Hier das lokal in Ollama geladene Modell eintragen
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Fehler bei der Kommunikation mit dem LLM: {e}")
        return "Sorry, I could not reach my brain right now."


def speak_on_reachy(mini: ReachyMini, text: str) -> None:
    """Erzeugt Audio mit Kokoro-TTS und spielt es über die Lautsprecher des Reachy Mini ab."""
    print("Erzeuge Sprachausgabe mit Kokoro-TTS...")
    samples, sample_rate = tts.create(text, voice=KOKORO_VOICE, speed=1.0, lang="en-us")

    # Als temporäre WAV-Datei speichern, da `media.play_sound()` einen Dateipfad erwartet
    # (GStreamer liest die Samplerate direkt aus dem WAV-Header aus).
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        wav_path = tmp_file.name
    try:
        sf.write(wav_path, samples, sample_rate)
        mini.media.play_sound(wav_path)
        # play_sound() kehrt sofort zurück, daher warten wir ungefähr die Audiolänge ab,
        # bevor die nächste Runde beginnt bzw. die Datei gelöscht wird.
        time.sleep(len(samples) / sample_rate + 0.5)
    finally:
        os.remove(wav_path)


def main():
    print("=== Reachy Mini Voice-Chat (Whisper -> LLM -> Kokoro-TTS) ===")
    print("Drücke ENTER für eine Spracheingabe, oder 'q' + ENTER zum Beenden.")

    print("Verbinde mit dem Reachy Mini...")
    with ReachyMini() as mini:
        while True:
            user_input = input("\n[ENTER] = sprechen, 'q' = beenden: ").strip().lower()
            if user_input == "q":
                break

            command_text = record_and_transcribe(duration=5.0)
            if not command_text:
                print("Keine Sprache erkannt, bitte erneut versuchen.")
                continue

            print("\n--- Erkannter Sprachbefehl ---")
            print(f"» {command_text} «")
            print("------------------------------")

            print("Frage lokales LLM um Rat...")
            answer = ask_llm(command_text)
            print(f"[LLM] {answer}")

            speak_on_reachy(mini, answer)

    print("Auf Wiedersehen!")


if __name__ == "__main__":
    main()
