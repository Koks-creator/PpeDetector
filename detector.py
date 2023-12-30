from dataclasses import dataclass
from typing import Union, Tuple
import pandas as pd
import torch
import numpy as np


@dataclass
class DetectorYoloV5:
    model_path: str
    conf_threshold: float = .2
    ultralitycs_path: str = "ultralytics/yolov5"
    model_type: str = "custom"
    force_reload: bool = True

    def __post_init__(self) -> None:
        self.model = torch.hub.load(self.ultralitycs_path, self.model_type, self.model_path, self.force_reload)
        self.model.conf = self.conf_threshold

    def detect(self, image: Union[str, np.array]) -> Tuple[np.array, pd.DataFrame]:
        """
        :param image: when using cv2 convert image to RGB
        :return:
        """
        results = self.model([image])

        return np.squeeze(results.render()), results.pandas().xyxy[0]
