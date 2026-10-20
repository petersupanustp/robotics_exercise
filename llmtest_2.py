import json
import pyexpat  # noqa: F401  # must be loaded before reachy_mini (GStreamer ships a conflicting libexpat.dll)
import numpy as np
from openai import OpenAI
from reachy_mini import ReachyMini

# 1. Verbindung zur lokalen Ollama-Instanz via OpenAI-API herstellen
# Ollama stellt standardmäßig unter Port 11434 eine OpenAI-kompatible Schnittstelle bereit.
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ein API-Key wird von Ollama nicht benötigt, darf aber nicht leer sein
)

def get_robot_action_from_llm(user_command: str) -> dict:
    """Sendet den Textbefehl an das lokale LLM und fordert eine strukturierte JSON-Antwort an."""

    # Der System-Prompt zwingt das LLM dazu, ausschließlich valides JSON auszugeben.
    system_prompt = (
        "Du bist das Gehirn des humanoiden Roboters Reachy Mini.\n"
        "Deine Aufgabe ist es, den Benutzerbefehl zu analysieren und eine Bewegung auszuwählen.\n"
        "Reagiere NUR mit einem JSON-Objekt im folgenden Format:\n"
        "{\n"
        "  \"bewegung\": \"hello\" | \"say_no\" | \"look_around\",\n"
        "  \"begruendung\": \"Kurze Erklärung deiner Wahl\"\n"
        "}"
    )

    try:
        response = client.chat.completions.create(
            model="phi4-mini",  # Hier das lokal in Ollama geladene Modell eintragen
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_command}
            ],
            # Zwingt kompatible Modelle (wie Llama 3 oder Qwen), direkt im JSON-Format zu antworten
            response_format={"type": "json_object"},
            temperature=0.0  # Niedrige Temperatur für deterministische Ergebnisse
        )

        # Antwort-Text in ein Python-Dictionary parsen
        res_json = json.loads(response.choices[0].message.content)
        return res_json

    except Exception as e:
        print(f"Fehler bei der Kommunikation mit dem LLM: {e}")
        return {"bewegung": "unknown", "begruendung": "Fehler"}

def execute_action(mini: ReachyMini, action: str):
    print(f"Führe Aktion aus: {action}")
    if action == "hello":
        mini.look_at_world(0.5, 0.0, 0.2, duration=1.0)
        mini.goto_target(antennas=np.deg2rad([45, -45]), duration=0.5)
        mini.goto_target(antennas=[0.0, 0.0], duration=0.5)
    elif action == "say_no":
        for y in (-0.3, 0.3, -0.3, 0.0):  # y = links (+) / rechts (-)
            mini.look_at_world(0.5, y, 0.0, duration=0.4)
    elif action == "look_around":
        for y in (-0.5, 0.5, 0.0):
            mini.look_at_world(0.5, y, 0.0, duration=1.0)
    else:
        print("Unbekannte oder keine Bewegung erkannt.")

def main():
    print("Verbinde mit dem Reachy Mini...")
    with ReachyMini() as mini:
        print("Gib einen Sprachbefehl ein (oder 'exit'/'quit' zum Beenden).")
        while True:
            user_voice_input = input("\nSprachbefehl: ").strip()

            if not user_voice_input:
                continue
            if user_voice_input.lower() in ("exit", "quit"):
                print("Beende...")
                break

            print("Frage lokales LLM um Rat...")
            llm_decision = get_robot_action_from_llm(user_voice_input)

            print("\n--- LLM Antwort ---")
            print(json.dumps(llm_decision, indent=2, ensure_ascii=False))
            print("-------------------\n")

            execute_action(mini, llm_decision.get("bewegung"))

if __name__ == "__main__":
    main()
