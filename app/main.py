
import cv2
import argparse
import time

from pathlib import Path

from app.detection.detector import SentinelDetector
from app.intelligence.zone_detector import ZoneDetector
from app.storage.database import initialize_database, save_event

from app.config import SNAPSHOTS_DIR


def run_sentinel(source):

    print("\n================================")
    print("       SENTINELCV v1.0")
    print("  Intelligent CCTV Monitoring")
    print("================================\n")

    initialize_database()

    detector = SentinelDetector()
    zone_detector = ZoneDetector()

    # Open camera or video
    cap = cv2.VideoCapture(source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    previous_inside = set()

    frame_count = 0
    start_time = time.perf_counter()
    fps = 0.0

    try:

        while True:

            success, frame = cap.read()

            if not success:
                print("[SentinelCV] Video stream ended.")
                break

            frame_count += 1

            height, width = frame.shape[:2]

            if zone_detector.zone is None:
                zone_detector.create_zone(width, height)

            # AI detection and object tracking
            detections = detector.process(frame)

            current_inside = set()

            # Draw restricted zone
            output = frame.copy()

            zone_detector.draw(output)

            for detection in detections:

                x1, y1, x2, y2 = detection["box"]

                center = detection["center"]

                class_name = detection["class_name"]
                confidence = detection["confidence"]
                track_id = detection["track_id"]

                inside = zone_detector.is_inside(center)

                if inside and track_id is not None:
                    current_inside.add(track_id)

                color = (0, 0, 255) if inside else (0, 255, 0)

                # Bounding box
                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                label = f"{class_name} {confidence:.2f}"

                if track_id is not None:
                    label += f" ID:{track_id}"

                cv2.putText(
                    output,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

                # Detect new zone entry
                if (
                    inside
                    and track_id is not None
                    and track_id not in previous_inside
                ):

                    timestamp = time.strftime("%Y%m%d_%H%M%S")

                    filename = (
                        f"intrusion_{timestamp}_"
                        f"{track_id}.jpg"
                    )

                    snapshot = SNAPSHOTS_DIR / filename

                    cv2.imwrite(str(snapshot), output)

                    save_event(
                        camera_id=str(source),
                        event_type="restricted_zone_intrusion",
                        object_type=class_name,
                        track_id=track_id,
                        confidence=confidence,
                        snapshot=snapshot
                    )

                    print(
                        f"[ALERT] {class_name} "
                        f"entered restricted zone. "
                        f"Tracking ID: {track_id}"
                    )

            previous_inside = current_inside

            # Performance monitoring
            elapsed = time.perf_counter() - start_time

            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.perf_counter()

            cv2.putText(
                output,
                f"SentinelCV | FPS: {fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                output,
                f"Objects: {len(detections)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imshow("SentinelCV - CCTV Monitoring", output)

            # Press Q to stop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        cap.release()
        cv2.destroyAllWindows()

        print("\n[SentinelCV] Monitoring stopped.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="0",
        help="Camera index or video file path"
    )

    args = parser.parse_args()

    source = (
        int(args.source)
        if args.source.isdigit()
        else args.source
    )

    run_sentinel(source)
