\# RiskAndIntentBased-Crosswalk



This project presents a computer vision–based system for pedestrian–vehicle risk analysis at crosswalks using YOLOv8 and ROI-based processing.



The system detects pedestrians and vehicles in real-time video streams, focuses on predefined crosswalk regions, and estimates risk levels based on spatial proximity.



---



\## 🚀 Features

\- Object detection using YOLOv8 (person, car, bus, truck, motorcycle)

\- ROI-based crosswalk localization

\- Risk level estimation: \*\*LOW / MEDIUM / HIGH\*\*

\- Scenario-based evaluation with multiple video conditions



---



\## 🧠 Model

\- \*\*YOLOv8s\*\* pretrained model (Ultralytics)

\- Single-stage real-time object detection

\- Optimized for accuracy–performance balance



> Model weights are automatically downloaded by Ultralytics and are not included in this repository.



---



\## 📂 Project Structure



RiskAndIntentBased-Crosswalk/

│

├── src/

│ └── main.py

│

├── data/

│ └── videos/

│ └── .gitkeep

│

├── outputs/

│ └── roi\_\*.json

│

├── .gitignore

└── README.md







---



\## 🎥 Dataset (Videos)



Due to GitHub file size limitations, the video files used in this project are **not included** in the repository.

You can download all test videos from the following Google Drive folder:

🔗 **Google Drive (Video Dataset):**  
https://drive.google.com/drive/folders/1XEcluy4vAJ05E8eZavThJgCwZ_vpOTUh

### Included video scenarios:
- `v1_basic_crosswalk.mp4` – Standard pedestrian crossing scenario
- `v2_waiting_pedestrian.mp4` – Pedestrians waiting at the crosswalk (LOW risk)
- `v3_vehicle_dense.mp4` – Dense vehicle traffic near the crosswalk
- `v4_risky_crossing.mp4` – Close pedestrian–vehicle interaction (HIGH risk)
- `v5_different_angle_light.mp4` – Different camera angle and low-light conditions

### After downloading:
Place the videos under the following directory structure:

```text
data/
└── videos/
    ├── v1_basic_crosswalk.mp4
    ├── v2_waiting_pedestrian.mp4
    ├── v3_vehicle_dense.mp4
    ├── v4_risky_crossing.mp4
    └── v5_different_angle_light.mp4






---



\## ▶️ How to Run



\### 1. Create virtual environment

```bash

python -m venv .venv

2\. Activate environment

Windows



bash


.venv\\Scripts\\activate

3\. Install dependencies

bash


pip install ultralytics opencv-python numpy

4\. Run the system

bash

python src/main.py --source data/videos/v1\_basic\_crosswalk.mp4



👩‍💻 Author

Kubra Adam

Software Engineering Student

Computer Vision \& Machine Learning Projects

