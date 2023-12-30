from time import time
from typing import List, Tuple, Union
from dataclasses import dataclass, field
import cv2
import numpy as np
import pandas as pd

from GigaBHP.detector import DetectorYoloV5
from GigaBHP.sortalg import Sort
from GigaBHP.config import config


@dataclass
class PpeApp:
    model_path: str = rf"{config.MODEL_FOLDER}/{config.MODEL_NAME}"
    conf_threshold: float = config.CONF_THRESHOLD
    ultralitycs_path: str = "ultralytics/yolov5"
    model_type: str = "custom"
    force_reload: bool = True
    sort_max_age: int = config.SORT_MAX_AGE
    classes: List[str] = field(default_factory=lambda: config.CLASSES)
    classes_to_track: List[str] = field(default_factory=lambda: config.CLASSES_TO_TRACK)
    debug_draw: bool = False
    text_color = (255, 255, 255)
    text_size = .7
    text_thickness = 2

    def __post_init__(self) -> None:
        self.detector = DetectorYoloV5(model_path=self.model_path, conf_threshold=self.conf_threshold)
        self.sort_tool = Sort(max_age=self.sort_max_age)

    @staticmethod
    def setup_detections_storage(classes_list: List[str], max_len: int) -> dict:
        detections_dict = {}
        for class_name in classes_list:
            detections_dict[class_name] = [[] for _ in range(max_len)]

        return detections_dict

    def get_detections(self, frame: np.array) -> Tuple[np.array, pd.DataFrame, List[int]]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_draw, detections = self.detector.detect(image=frame_rgb)
        class_counts = detections["name"].value_counts().to_list()

        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)

        return img_draw, detections, class_counts

    def update_tracker_and_detection_info(self, frame: np.array, detections: pd.DataFrame, detection_info: dict,
                                          classes_to_track: List[str]):
        detections_ar = np.empty((0, 5))

        for row in detections.iterrows():
            detection = row[1]

            bbox = (int(detection.xmin), int(detection.ymin), int(detection.xmax), int(detection.ymax))
            conf = detection.confidence
            class_name = detection['name']

            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            if class_name.lower() in classes_to_track:
                curr_arr = np.array([x1, y1, x2, y2, conf])
                detections_ar = np.vstack((detections_ar, curr_arr))

            index = len(detection_info[class_name]) - detection_info[class_name].count([])

            detection_info[class_name][index] = (x1, y1, x2, y2)
            if self.debug_draw:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 225), 2)

        tracker_results = self.sort_tool.update(detections_ar)

        return frame, detection_info, tracker_results

    @staticmethod
    def get_areas(x1: int, y1: int, x2: int, y2: int) -> Tuple[np.array, np.array]:
        helmet_area = np.array([
            [x1, y1],
            [x2, y1],
            [x2, int(y1 + ((abs(y1 - y2)) // 3))],
            [x1, int(y1 + ((abs(y1 - y2)) // 3))],
        ], np.int32)

        vest_area = np.array([
            [x1, int(y1 + ((abs(y1 - y2)) // 3))],
            [x2, int(y1 + ((abs(y1 - y2)) // 3))],
            [x2, y2],
            [x1, y2],
        ], np.int32)

        return helmet_area, vest_area

    @staticmethod
    def get_center(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
        return int(x1 + ((abs(x1-x2))//2)), int(y1 + ((abs(y1-y2))//2))

    def on_image(self, frame: np.array) -> Tuple[np.array, np.array, np.array]:
        org_frame = frame.copy()
        img_draw, detections, class_counts = self.get_detections(frame=frame)
        if class_counts:
            max_len = max(class_counts)
            detection_info = self.setup_detections_storage(classes_list=self.classes, max_len=max_len)
            frame, detection_info, tracker_results = self.update_tracker_and_detection_info(
                frame=frame, detections=detections, detection_info=detection_info, classes_to_track=self.classes_to_track,
            )

            for res in tracker_results:
                p_x1, p_y1, p_x2, p_y2, person_id = int(res[0]), int(res[1]), int(res[2]), int(res[3]), int(res[4])
                helmet_area, vest_area = self.get_areas(p_x1, p_y1, p_x2, p_y2)

                if self.debug_draw:
                    cv2.drawContours(frame, [helmet_area], -1, 255, 2)
                    cv2.drawContours(frame, [vest_area], -1, (0, 0, 0), 2)

                is_helmet = False
                is_vest = False
                for helmet, vest in zip(detection_info["helmet"], detection_info["vest"]):
                    if helmet:
                        h_x1, h_y1, h_x2, h_y2 = helmet
                        helmet_center = self.get_center(h_x1, h_y1, h_x2, h_y2)
                        result = cv2.pointPolygonTest(helmet_area, helmet_center, False)

                        if result == 1:
                            is_helmet = True

                        if self.debug_draw:
                            cv2.circle(frame, helmet_center, 5, (255, 0, 0), -1)

                    if vest:
                        v_x1, v_y1, v_x2, v_y2 = vest
                        vest_center = self.get_center(v_x1, v_y1, v_x2, v_y2)
                        result = cv2.pointPolygonTest(vest_area, vest_center, False)

                        if result == 1:
                            is_vest = True

                        if self.debug_draw:
                            cv2.circle(frame, vest_center, 5, (255, 0, 0), -1)

                if all([is_vest, is_helmet]):
                    color = (0, 200, 0)
                else:
                    color = (0, 0, 200)

                cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), color, 2)

                cv2.rectangle(frame, (p_x1, p_y1), (p_x1 + 150, p_y1 + 65), color, -1)
                cv2.putText(frame, f"Id: {person_id}", (p_x1 + 3, p_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, self.text_size,
                            self.text_color, self.text_thickness)
                cv2.putText(frame, f"Helmet: {is_helmet}", (p_x1 + 3, p_y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, self.text_size,
                            self.text_color, self.text_thickness)
                cv2.putText(frame, f"Vest: {is_vest}", (p_x1 + 3, p_y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, self.text_size,
                            self.text_color, self.text_thickness)
        return frame, img_draw, org_frame

    def main(self, cap_source: Union[str, int]):
        cap = cv2.VideoCapture(cap_source)

        p_time = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("No frames, exiting...")
                break

            frame, img_draw, org_frame = self.on_image(frame=frame)

            c_time = time()
            fps = int(1 / (c_time - p_time))
            p_time = c_time

            cv2.putText(frame, f"FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow("res", frame)
            if self.debug_draw:
                cv2.imshow("res2", img_draw)
            key = cv2.waitKey(1)
            if key == 27:
                print("Exiting")
                break


if __name__ == '__main__':
    ppe = PpeApp()
    ppe.main(cap_source=rf"{config.VIDEOS_FOLDER}/{config.VIDEO_NAME}")

