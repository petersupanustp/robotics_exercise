import numpy as np
import sounddevice as sd
import whisper

# 1. Laden des lokalen Whisper-Modells
# Für Bildungszwecke auf Standard-Laptops ist "base" der perfekte Kompromiss 
# aus Geschwindigkeit (Echtzeit) und Genauigkeit (auch auf Deutsch).
print("Lade lokales Whisper-Modell ('base')...")
whisper_model = whisper.load_model("base")

def record_audio(duration=5, sample_rate=16000) -> str:
    """
    Nimmt für eine festgelegte Dauer Audio über das Standard-Mikrofon auf
    und gibt den transkribierten Text zurück.
    """
    print(f"\n[Miko] Aufnahme startet für {duration} Sekunden... Bitte sprechen!")
    
    # Aufnahme starten (1 Kanal, da Mono für Whisper völlig ausreicht)
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Warten, bis die Aufnahmezeit abgelaufen ist
    print("[Mikro] Aufnahme beendet. Verarbeite Audio lokal...")

    # Whisper akzeptiert direkt ein float32-Mono-Array mit 16 kHz. So wird keine WAV-Datei
    # und kein externes ffmpeg (nötig beim Laden von Dateien) benötigt.
    audio = np.asarray(audio_data, dtype=np.float32).flatten()

    # 2. Lokale Transkription mit Whisper durchführen
    # 'language="de"' beschleunigt die Erkennung und verhindert, dass Whisper versucht, ins Englische zu übersetzen.
    result = whisper_model.transcribe(audio, language="de")
    
    transcribed_text = result["text"].strip()
    return transcribed_text

def main():
    print("=== Reachy Mini Voice-Control Modul ===")
    input("Drücke ENTER, um die Spracheingabe zu starten...")
    
    # 5 Sekunden Aufnahmezeit einräumen
    befehl = record_audio(duration=5)
    
    print("\n--- Erkannter Sprachbefehl ---")
    print(f"» {befehl} «")
    print("------------------------------")
    
    # Hier folgt im nächsten Schritt der Aufruf deines LLMs:
    # llm_decision = get_robot_action_from_llm(befehl)

if __name__ == "__main__":
    main()
