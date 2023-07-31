from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import time
from shapely.geometry import Polygon, Point


class YOLOVideoProcessor:
    def __init__(self, model_weights_path, video_path):
        self.model = YOLO(model_weights_path)
        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                            "umbrella",
                            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                            "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
                            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        # Alt bölgenin poligonunu oluştururuz, kişilerin sayılacağı bölge burası
        self.lower_region_polygon = Polygon([(0, 540), (1920, 540), (1920, 1080), (0, 1080)])
        # Alt bölgeye giren kişilerin id'lerini tutmak için boş bir küme oluştururuz.
        self.first_in = set()
        # Metin yazmak için kullanılacak yazı tipini belirleriz.
        self.font = cv2.FONT_HERSHEY_DUPLEX
        # FPS hesaplamak için kullanılacak değişkenler. Her karedeki süre farkını hesaplamak için kullanılır.
        self.prev_frame_time = 0
        self.new_frame_time = 0

    def detect_objects(self):
        while True:
            # FPS'i hesaplaıyoruz.
            self.new_frame_time = time.time()
            fps = 1 / (self.new_frame_time - self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time
            print(fps)
            success, frame = self.cap.read()
            if not success:
                break
            # Resmi RGB uzayına dönüştürdük
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model.track(rgb_img, persist=True, verbose=False)
            detected_objects = self._get_detected_objects(results)
            self.counter(detected_objects, frame)
            self.draw_bounding_boxes(frame, detected_objects)
            self.first_in.clear()

            cv2.imshow('Detected Objects', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def _get_detected_objects(self, results):
        detected_objects = []
        for r in results:
            r = r.boxes
            for box in r:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cl = int(box.cls[0])
                ids = box.id[0]
                cls = int(box.cls[0])
                class_name = self.class_names[cl]
                detected_objects.append({
                    'class_name': class_name,
                    'confidence': conf,
                    'bounding_box': (x1, y1, x2 - x1, y2 - y1),
                    'id': ids,
                    'cls id': cls
                })

        return detected_objects

    def draw_bounding_boxes(self, frame, detected_objects):#Tespit edilen insanların etrafını cizen fonksiyon
        for obj in detected_objects:
            x1, y1, w, h = obj['bounding_box']
            cls = obj['cls id']
            if cls == 0:
                cvzone.cornerRect(frame, (x1, y1, w, h))
                cvzone.putTextRect(frame, f'{obj["class_name"]}{obj["confidence"]}-id:{obj["id"]}', (x1, y1 - 10),
                               thickness=1, scale=1)
        #Ekranin alt yarısındaki insan sayısını yazdırdım
        cv2.putText(frame, f'People count: {len(self.first_in)}', (10, 50), self.font, 1, (255, 255, 255), 2)
        #Ekranin alt yarısındaki insanların id'lerini yazdırdım
        cv2.putText(frame, f'Poeple id: {(self.first_in)}', (10, 80), self.font, 1, (255, 255, 255), 2)
        # Çizgiyi ekrana çiziyor.
        cv2.line(frame, (0, 540), (1920, 540), (255, 255, 255), 2)

    def counter(self, detected_objects, frame):
        for obj in detected_objects:
            x1, y1, x2, y2 = obj['bounding_box']
            score = obj['confidence']
            cls = obj['cls id']
            ids = obj.get('id', None)
            # Değerleri uygun bir fomata çevirdim.
            x1, y1, x2, y2, score, cls, ids = int(x1), int(y1), int(x2), int(y2), float(score), int(cls), int(ids)
            # burada 0.5' lik bir threshold uyguladık. Tespit edilme değeri daha küçükse değerlendirmeye almıyoruz.
            if score < 0.5:
                continue
            # İnsan sayacağımiz için insan dışındaki diğer nesneleri değerlendirmeye almıyoruz.
            if cls != 0:
                continue
            person_polygon = Polygon([(x1, y1), (x1 + x2, y1), (x1 + x2, y1 + y2), (x1, y1 + y2)])
            if person_polygon.intersects(self.lower_region_polygon):
                # Kişi Poligonunun Merkez noktasını belirledim
                central_x = (x1 + x1 + x2) / 2
                central_y = (y1 + y1 + y2) / 2
                central_point = Point(central_x, central_y)
                # Merkez nokta alt bölge ile kesişiyorsa, bu kişiyi sayın ve merkez noktasını işaretle
                if central_point.intersects(self.lower_region_polygon):
                    cv2.circle(frame, (int(central_x), int(central_y)), 4, (0, 255, 255), -1)
                    self.first_in.add(ids)

if __name__ == "__main__":
    model_weights_path = '../Yolo-Weights/yolov8n.pt'
    video_path = 'Videos/people.mp4'

    video_processor = YOLOVideoProcessor(model_weights_path, video_path)
    video_processor.detect_objects()
