from transformers import pipeline
from PIL import Image

# Initialisiert die lokale Pipeline. Lädt das Modell beim ersten Mal automatisch herunter.
# Ein exzellentes Modell für diesen Zweck:
classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

# Bild laden
image = Image.open("gesicht.png")

# Vorhersage lokal ausführen
predictions = classifier(image)

print("Erkannte Stimmungen (lokal berechnet):")
for pred in predictions:
    # Der Score wird in Prozent umgerechnet
    print(f"- {pred['label']}: {pred['score'] * 100:.2f}%")
