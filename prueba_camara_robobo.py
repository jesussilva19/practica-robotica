from robobopy_videostream.RoboboVideo import RoboboVideo
from robobopy.Robobo import Robobo
from ultralytics import YOLO
import cv2
import signal
import sys

IP = "192.168.x.x"  # IP del móvil con la app del Robobo

rob = Robobo(IP)
video = RoboboVideo(IP)
model = YOLO("yolo11n.pt")  # detección normal

def cleanup():
    try: video.disconnect()
    except: pass
    try: rob.disconnect()
    except: pass
    cv2.destroyAllWindows()
    print("Exit cleanly.")
    sys.exit(0)

signal.signal(signal.SIGINT, lambda s,f: cleanup())

def main():
    rob.connect()
    video.connect()
    rob.startStream()

    while True:
        frame, ts, sync_id, frame_id = video.getImageWithMetadata()

        results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("Robobo camera + YOLO", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cleanup()

if __name__ == "__main__":
    main()
