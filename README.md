# YOLO Video Processor

This Python script utilizes YOLO (You Only Look Once) for object detection in a video stream. It detects and tracks objects, focusing primarily on people within a specified region of interest.

## Dependencies
- [Ultralytics YOLO](https://github.com/ultralytics/yolov5): YOLO implementation for object detection.
- [OpenCV](https://opencv.org/): Open Source Computer Vision Library.
- [CVZone](https://github.com/cvzone/cvzone): Python library to make working with OpenCV easier.

## Installation
1. Install the necessary packages:
    ```bash
    pip install opencv-python cvzone numpy shapely ultralytics
    ```
2. Download YOLOv5 model weights from [here](https://github.com/ultralytics/yolov5/releases) and place them in a directory accessible from your script.

## Usage
1. Update the `model_weights_path` and `video_path` variables in the script to point to your YOLO model weights file and the video file you want to process, respectively.
2. Run the script:
    ```bash
    python yolo_video_processor.py
    ```
3. The script will display the video with bounding boxes drawn around detected objects. Press 'q' to exit the video stream.

## Features
- Detects and tracks multiple classes of objects using YOLO.
- Specifically counts and tracks people within a defined region of interest.
- Draws bounding boxes around detected objects and displays the count of people in the lower region of the screen.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request on the GitHub repository.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
