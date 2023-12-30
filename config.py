from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    MODEL_FOLDER: str = "Model"
    MODEL_NAME: str = "best.pt"
    CONF_THRESHOLD: float = .3
    TEST_IMAGES: str = "TestImages"
    VIDEOS_FOLDER: str = "Videos"
    VIDEO_NAME: str = "pexels-mikael-blomkvist-8964295 (720p).mp4"
    CLASSES: List[str] = field(default_factory=lambda: ["person", "helmet", "vest"])
    CLASSES_TO_TRACK: List[str] = field(default_factory=lambda: ["person"])
    SORT_MAX_AGE: int = 20


config = Config()
