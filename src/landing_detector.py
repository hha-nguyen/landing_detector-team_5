from ultralytics import YOLO

class LandingDetector:

    def __init__(self):
        """
        You should hard fix all the requirement parameters
        """
        self.name = 'LandingDetector'

    def detect(self, img):
        modal = YOLO("")
        
        results = modal.train(data="config.yaml", epochs=1)
        return results