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



yaml

Kodu kopyala



---



\## 🎥 Dataset (Videos)



Due to GitHub size limitations, video files are \*\*not included\*\* in this repository.



Download the videos from:  

🔗 \*\*\[Google Drive link – add here]\*\*



After downloading, place the videos under:



data/videos/



yaml

Kodu kopyala



---



\## ▶️ How to Run



\### 1. Create virtual environment

```bash

python -m venv .venv

2\. Activate environment

Windows



bash

Kodu kopyala

.venv\\Scripts\\activate

3\. Install dependencies

bash

Kodu kopyala

pip install ultralytics opencv-python numpy

4\. Run the system

bash

Kodu kopyala

python src/main.py --source data/videos/v1\_basic\_crosswalk.mp4



👩‍💻 Author

Kubra Adam

Software Engineering Student

Computer Vision \& Machine Learning Projects

