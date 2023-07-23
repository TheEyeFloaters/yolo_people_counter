from ultralytics import YOLO
import cv2
import cvzone
import math

class YOLOVideoProcessor:
    def __init__(self, model_weights_path, video_path):
        self.model = YOLO(model_weights_path)
        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                            "baseball bat",
                            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                            "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                            "teddy bear", "hair drier", "toothbrush"
                            ]

    def detect_objects(self):#Videoda Nesne Tespiti yapmak icin YOLO modelini kullanan fonksiyon
        while True:
            success, img = self.cap.read()
            if not success:
                break

            results = self.model(img, stream=True)
            detected_objects = self._get_detected_objects(results)
            self.draw_bounding_boxes(img, detected_objects)

            cv2.imshow('Detected Objects', img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def _get_detected_objects(self, results):#Sonuclari donduren fonksiyon
        detected_objects = []
        for r in results:
            r = r.boxes
            for box in r:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cl = int(box.cls[0])
                class_name = self.class_names[cl]
                detected_objects.append({
                    'class_name': class_name,
                    'confidence': conf,
                    'bounding_box': (x1, y1, x2 - x1, y2 - y1)
                })
        return detected_objects

    def draw_bounding_boxes(self, img, detected_objects):#Tespit edilen nesnelerin etrafını cizen fonksiyon
        for obj in detected_objects:
            x1, y1, w, h = obj['bounding_box']
            cvzone.cornerRect(img, (x1, y1, w, h))
            cvzone.putTextRect(img, f'{obj["class_name"]}{obj["confidence"]}', (x1, y1 - 10), thickness=1, scale=1)


if __name__ == "__main__":
    model_weights_path = '../Yolo-Weights/yolov8n.pt'
    video_path = 'Videos/people.mp4'

    video_processor = YOLOVideoProcessor(model_weights_path, video_path)
    video_processor.detect_objects()
