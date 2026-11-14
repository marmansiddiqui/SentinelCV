
import cv2
import numpy as np


class ZoneDetector:

    def __init__(self):
        self.zone = None

    def create_zone(self, width, height):

        # Define restricted-zone coordinates.
        points = [
            (int(width * 0.55), int(height * 0.20)),
            (int(width * 0.95), int(height * 0.20)),
            (int(width * 0.95), int(height * 0.85)),
            (int(width * 0.55), int(height * 0.85))
        ]

        # Convert points into the format required by OpenCV.
        self.zone = np.array(
            points,
            dtype=np.int32
        ).reshape((-1, 1, 2))

    def is_inside(self, point):

        if self.zone is None:
            return False

        # OpenCV requires floating-point point coordinates.
        point = (
            float(point[0]),
            float(point[1])
        )

        return cv2.pointPolygonTest(
            self.zone,
            point,
            False
        ) >= 0

    def draw(self, frame):

        if self.zone is not None:

            # Draw restricted-zone polygon.
            cv2.polylines(
                frame,
                [self.zone],
                True,
                (0, 0, 255),
                2
            )

            # Add zone label.
            label_position = tuple(
                self.zone[0][0]
            )

            cv2.putText(
                frame,
                "RESTRICTED ZONE",
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        return frame
