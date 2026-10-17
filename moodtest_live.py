import cv2
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import pipeline

# Emotions-Klassifikator (Hugging Face transformers). Lädt das Modell beim ersten Mal automatisch herunter.
classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

# Gesichtsdetektor: Der Klassifikator bewertet nur ein Bild, findet aber keine Gesichter.
# OpenCV 5 enthält keine Haar-Cascades mehr, daher wird YuNet (ONNX) vom Hugging Face Hub verwendet.
yunet_path = hf_hub_download("opencv/face_detection_yunet", "face_detection_yunet_2023mar.onnx")
detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), score_threshold=0.8)

# 1. Verbindung zur Webcam herstellen (0 ist meistens die Standard-Kamera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden.")
    exit()

print("Starte Echtzeit-Erkennung. Drücke 'q' zum Beenden.")

while True:
    # Frame für Frame von der Webcam lesen
    ret, frame = cap.read()
    if not ret:
        print("Fehler beim Empfangen des Kamerabilds.")
        break

    frame_h, frame_w = frame.shape[:2]

    try:
        # Gesichter im aktuellen Frame finden
        detector.setInputSize((frame_w, frame_h))
        _, faces = detector.detect(frame)

        # Falls mehrere Gesichter im Bild sind, gehen wir sie einzeln durch
        for face in faces if faces is not None else []:
            # Koordinaten des Gesichts für das Rechteck auslesen (auf den Bildrand begrenzen)
            x, y, w, h = map(int, face[:4])
            x, y = max(x, 0), max(y, 0)
            w, h = min(w, frame_w - x), min(h, frame_h - y)
            if w <= 0 or h <= 0:
                continue

            # Gesichtsausschnitt (BGR -> RGB) an den transformers-Klassifikator übergeben
            face_rgb = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
            predictions = classifier(Image.fromarray(face_rgb), top_k=1)

            # Dominante Emotion ermitteln
            dominant_emotion = predictions[0]['label']
            score = predictions[0]['score']

            # 2. Visuelle Elemente in das Live-Bild einfügen
            # Blaues Rechteck um das Gesicht zeichnen (Dicke: 2 Pixel)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Text mit der Stimmung über das Rechteck schreiben
            cv2.putText(frame, f"{dominant_emotion} ({score * 100:.0f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    except Exception as e:
        # Fängt Fehler ab, falls bei der Analyse im Frame etwas schiefgeht
        print(f"Fehler bei der Analyse: {e}")

    # Das verarbeitete Live-Bild im Fenster anzeigen
    cv2.imshow('Echtzeit Stimmungs-Erkennung', frame)

    # Das Programm beenden, wenn die Taste 'q' gedrückt wird
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Nach dem Ende der Schleife alles sauber schließen und freigeben
cap.release()
cv2.destroyAllWindows()
