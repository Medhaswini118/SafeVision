# 🛡️ SafeVision – Detecting Safe Objects with YOLO

## 📌 Overview
SafeVision is an AI-powered application built with **YOLOv8** and **Streamlit** that detects **safe objects** in images.  
Unlike typical hazard detection systems, our model focuses on identifying objects that are **non-hazardous / safe**, ensuring workplace safety monitoring and awareness.  

🌐 Live Demo: [SafeVision on Streamlit](https://safevision.streamlit.app/)

---

## 👥 Team INFINIA
- **Team Leader**: Medhaswini Kambhampati  
- **Team Members**: Niharika Gubba, Pooja Singh  

---

## ⚙️ Project Workflow
1. **Dataset Preparation**
   - Dataset provided by Falcon AI
   - Images divided into **Train, Validation, and Test** sets
   - Labels in YOLO format

2. **Model Training**
   - Framework: Ultralytics YOLOv8
   - Environment: Anaconda (`EDU` environment)
   - Training Command:
     ```bash
     python train.py
     ```

3. **Evaluation Metrics**
   - mAP@50, Precision, Recall
   - Confusion Matrix and Predictions

4. **Deployment**
   - Deployed using **Streamlit Cloud**
   - Supports **image upload + sample images** for predictions

---

## 📊 Results & Performance
- **Highest mAP@50**: **59%**  
- **Precision**: **85%**  
- **Recall**: **50%**

📉 Training Curve (Loss & mAP):
![Training Curve](forreport/train4/outpage.png)

📈 Confusion Matrix (Test Set):
![Confusion Matrix](forreport/train4/confusion_matrix.png)

✅ Example Predictions:
| Test Image | Model Prediction |
|------------|------------------|
| ![Test1](forreport/train3/test_images/test1.png) | ![Pred1](forreport/train3/predictions/pred1.png) |
| ![Test2](forreport/train3/test_images/test2.png) | ![Pred2](forreport/train3/predictions/pred2.png) |

---

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/Medhaswini118/SafeVision.git
   cd SafeVision
2. Install dependencies:

pip install -r requirements.txt

3. Run the Streamlit app:

streamlit run app.py
