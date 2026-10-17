import threading
import time

import cv2
import numpy as np
from reachy_mini import ReachyMini
from ultralytics import YOLO

def head_movements(mini, stop_event):
    # look_at_world blockiert bis die Bewegung fertig ist, daher läuft diese Funktion
    # in einem eigenen Thread, damit die Kamera-Schleife parallel weiterläuft.
    time.sleep(5.0)  # kurz warten, bis das Videofenster läuft
    targets = [(0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.0)]
    while not stop_event.is_set():
        for x, y, z in targets:
            mini.look_at_world(x, y, z, duration=1.0)
        time.sleep(2.0)
    #mini.goto_target(antennas=np.deg2rad([45, -45]), duration=0.5)
    #mini.goto_target(antennas=[0.0, 0.0], duration=0.5)


def main():
    # 1. Laden des lokalen KI-Modells (YOLOv8 Nano - lädt sich beim ersten Mal autom. herunter)
    # Dieses Modell läuft problemlos in Echtzeit auf normalen Laptops (auch ohne dedizierte GPU).
    print("Lade lokales KI-Modell...")
    model = YOLO("yolov8n.pt")  

    print("Verbinde mit dem Reachy Mini (Simulation/Hardware)...")
    # Der Context Manager stellt sicher, dass Verbindungen beim Beenden sauber geschlossen werden.
    # 'media_backend="default"' wählt automatisch die beste Pipeline (Sim/WebRTC).
    with ReachyMini(media_backend="default") as mini:
        print("Kamera-Stream gestartet. Drücke 'q' im Videofenster zum Beenden.")

        stop_event = threading.Event()
        mover = threading.Thread(target=head_movements, args=(mini, stop_event), daemon=True)
        mover.start()

        while True:
            # 2. Frame aus dem robotereigenen Media-System abgreifen (gibt ein NumPy-Array zurück)
            frame = mini.media.get_frame()
            
            if frame is None:
                continue

            # Reachy Mini liefert das Bild standardmäßig im RGB-Format.
            # OpenCV benötigt für die Darstellung (imshow) typischerweise BGR.
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            bgr_frame = cv2.resize(frame, (640, 384))

            # 3. Das Bild an das lokale KI-Modell übergeben (Inferenz)
            # stream=True sorgt für optimierte Speichernutzung bei Video-Feeds
            results = model(bgr_frame, stream=True)

            # 4. Ergebnisse visualisieren (Bounding Boxes auf das Bild zeichnen)
            for r in results:
                annotated_frame = r.plot()  # Generiert ein Bild mit eingezeichneten Boxen

            # 5. Das verarbeitete Bild im Fenster anzeigen
            cv2.imshow("Reachy Mini AI Vision Feed", annotated_frame)

            # Abbruchbedingung: Wenn 'q' gedrückt wird, Schleife verlassen
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Bewegungs-Thread beenden, bevor die Verbindung geschlossen wird
        stop_event.set()
        mover.join(timeout=3.0)

    # Fenster schließen, wenn der Context Manager verlassen wurde
    cv2.destroyAllWindows()
    print("Stream sauber beendet.")

if __name__ == "__main__":
    main()
