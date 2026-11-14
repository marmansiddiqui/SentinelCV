import torch
import cv2

from ultralytics import YOLO

from app.config import (
    MODEL_NAME,
    IMAGE_SIZE,
    CONFIDENCE,
    DEVICE
)


class SentinelDetector:

    def __init__(self):

        torch.set_num_threads(4)
        cv2.setNumThreads(2)

        print("[SentinelCV] Loading detection model...")

        self.model = YOLO(MODEL_NAME)

        # COCO classes:
        # 0 = person
        # 2 = car
        # 3 = motorcycle
        # 5 = bus
        # 7 = truck
        self.classes = [0, 2, 3, 5, 7]

        # Lightweight Python tracker settings
        self.next_track_id = 1
        self.tracks = {}

        self.max_age = 20
        self.iou_threshold = 0.25

        print("[SentinelCV] Detection engine ready.")
        print("[SentinelCV] Python IoU tracker initialized.")

    @staticmethod
    def calculate_iou(box_a, box_b):

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)
        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)

        intersection_width = max(
            0, intersection_x2 - intersection_x1
        )

        intersection_height = max(
            0, intersection_y2 - intersection_y1
        )

        intersection_area = (
            intersection_width * intersection_height
        )

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union_area = area_a + area_b - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    def update_tracks(self, detections):

        # Mark existing tracks as not matched this frame.
        for track in self.tracks.values():
            track["missed"] += 1

        possible_matches = []

        # Calculate IoU between existing tracks and new detections.
        for track_id, track in self.tracks.items():

            for detection_index, detection in enumerate(detections):

                # Only match objects of the same class.
                if track["class_id"] != detection["class_id"]:
                    continue

                iou = self.calculate_iou(
                    track["box"],
                    detection["box"]
                )

                if iou >= self.iou_threshold:
                    possible_matches.append(
                        (iou, track_id, detection_index)
                    )

        # Highest IoU matches are considered first.
        possible_matches.sort(reverse=True)

        matched_tracks = set()
        matched_detections = set()

        for iou, track_id, detection_index in possible_matches:

            if track_id in matched_tracks:
                continue

            if detection_index in matched_detections:
                continue

            detection = detections[detection_index]

            # Update existing track.
            self.tracks[track_id]["box"] = detection["box"]
            self.tracks[track_id]["class_id"] = detection["class_id"]
            self.tracks[track_id]["missed"] = 0

            detection["track_id"] = track_id

            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        # Create new tracks for unmatched detections.
        for detection_index, detection in enumerate(detections):

            if detection_index in matched_detections:
                continue

            track_id = self.next_track_id
            self.next_track_id += 1

            detection["track_id"] = track_id

            self.tracks[track_id] = {
                "box": detection["box"],
                "class_id": detection["class_id"],
                "missed": 0
            }

        # Remove tracks that have disappeared for too long.
        expired_tracks = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missed"] > self.max_age
        ]

        for track_id in expired_tracks:
            del self.tracks[track_id]

        return detections

    def process(self, frame):

        # Run YOLO detection only. No Ultralytics tracking or lap DLL.
        results = self.model.predict(
            source=frame,
            classes=self.classes,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        detections = []

        if result.boxes is not None:

            boxes = result.boxes

            for i in range(len(boxes)):

                coordinates = boxes.xyxy[i].cpu().numpy()

                class_id = int(boxes.cls[i].item())

                confidence = float(boxes.conf[i].item())

                x1, y1, x2, y2 = map(int, coordinates)

                detections.append({
                    "class_id": class_id,
                    "class_name": self.model.names[class_id],
                    "confidence": confidence,
                    "track_id": None,
                    "box": (x1, y1, x2, y2),
                    "center": (
                        (x1 + x2) // 2,
                        (y1 + y2) // 2
                    )
                })

        # Assign persistent IDs using our Python tracker.
        detections = self.update_tracks(detections)

        return detections