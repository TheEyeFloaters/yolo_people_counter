YOLO Video Processor README
This Python script utilizes YOLO (You Only Look Once) for object detection in a video stream. It detects and tracks objects, focusing primarily on people within a specified region of interest.

Dependencies:
Ultralytics YOLO: YOLO implementation for object detection.
OpenCV: Open Source Computer Vision Library.
CVZone: Python library to make working with OpenCV easier.
Installation:
Install the necessary packages:
bash
Copy code
pip install opencv-python cvzone numpy shapely ultralytics
Download YOLOv5 model weights from here and place them in a directory accessible from your script.
Usage:
Update the model_weights_path and video_path variables in the script to point to your YOLO model weights file and the video file you want to process, respectively.

Run the script:

bash
Copy code
python yolo_video_processor.py
The script will display the video with bounding boxes drawn around detected objects. Press 'q' to exit the video stream.
Features:
Detects and tracks multiple classes of objects using YOLO.
Specifically counts and tracks people within a defined region of interest.
Draws bounding boxes around detected objects and displays the count of people in the lower region of the screen.
Contributing:
Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request on the GitHub repository.

License:
This project is licensed under the MIT License - see the LICENSE file for details.
