# 🏀 AI-Powered Basketball Analysis System

An end-to-end Computer Vision pipeline designed to analyze basketball gameplay. This system leverages state-of-the-art Deep Learning models to track players and the ball, identify teams, and project game action onto a 2D tactical map.

---

## 📺 Demo
 


---

## 🌟 Key Features
* **Object Detection & Tracking:** Utilizes fine-tuned **YOLO** models to detect players, referees, and the basketball with high precision across frames.
* **Team Assignment:** Automated team classification using **Zero-Shot Image Classification** (CLIP-based) to identify jersey colors without manual labeling.
* **Game Intelligence:**
    * **Ball Possession:** Real-time tracking of which player or team has control of the ball.
    * **Event Detection:** Automated logging of **Passes** and **Interceptions** by analyzing ball trajectory and possession transitions.
    * **Possession Analytics:** Calculates ball acquisition percentages for each team throughout the game.
* **Tactical Analysis:**
    * **Court Keypoint Detection:** Identification of specific court landmarks for spatial orientation.
    * **Perspective Transformation:** Translates the 3D broadcast camera view into a **2D Top-Down Tactical Map** using homography, providing a clean "birds-eye" view of player positioning.

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **Computer Vision:** OpenCV, Ultralytics (YOLO)
- **Deep Learning:** PyTorch, Hugging Face (Transformers for Zero-Shot)
- **Object Tracking:** ByteTrack / Supervision

## 📊 Methodology
1. **Inference:** Processing video frames through fine-tuned YOLO models for robust detection under varying lighting and camera angles.
2. **Feature Extraction:** Cropping player images and passing them through a Zero-Shot Classifier to assign team labels based on visual features.
3. **Geometry:** Using a custom-trained keypoint model to find court corners, followed by a perspective transform to map pixel data to a normalized 2D coordinate system.
4. **Analysis:** Implementing custom logic to detect interactions between the "Ball" and "Player" classes to determine game events.
