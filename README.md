---

## **Traffic Sign Recognition Using Deep Learning**

### **Overview**
This project implements a deep learning model to recognize German traffic signs using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. The application is deployed as an interactive web app using **Streamlit**, allowing users to upload traffic sign images and receive real-time predictions.

---

### **Objective**
The primary goal of this project is to create a robust traffic sign recognition system that can:
- Accurately classify 43 different traffic sign classes.
- Provide real-time inference through a user-friendly Streamlit interface.
- Serve as a foundation for intelligent transportation systems and self-driving cars.

---

### **Dataset**
The dataset used in this project is the **German Traffic Sign Recognition Benchmark (GTSRB)**. It consists of over 50,000 images of traffic signs captured under various conditions.

#### **Key Details**:
- **Training Set**: 34,799 images
- **Validation Set**: 4,410 images
- **Test Set**: 12,630 images
- **Image Dimensions**: 32 x 32 x 3 (RGB)
- **Number of Classes**: 43

#### **Class Distribution Visualization**:
A bar chart was generated to visualize the number of samples per class, highlighting class imbalances.

---

### **Model Architecture**
#### **Pre-trained MobileNet v2**
To improve accuracy and efficiency, the project uses a pre-trained MobileNet v2 model fine-tuned on the GTSRB dataset. MobileNet v2 is a lightweight convolutional neural network optimized for mobile and embedded devices.

#### **Key Features**:
- Depthwise Separable Convolutions for reduced computation.
- Modified classifier layer for 43 traffic sign classes.
- Fine-tuned using transfer learning techniques.

---

### **Methodology**
1. **Data Preprocessing**:
   - Resized all images to `32x32`.
   - Normalized pixel values to `[1]` using mean `(0.5, 0.5, 0.5)` and standard deviation `(0.5, 0.5, 0.5)`.
   - Applied data augmentation techniques like random rotations and flips.

2. **Model Training**:
   - Fine-tuned MobileNet v2 with frozen feature extraction layers.
   - Optimized using Adam optimizer with a learning rate of `0.001`.
   - Trained for `10 epochs` with batch size `32`.

3. **Evaluation Metrics**:
   - Accuracy
   - Precision
   - Recall
   - F1-score

4. **Deployment**:
   - Deployed as a Streamlit app on Streamlit Cloud.
   - Interactive interface allows users to upload images and receive predictions in real-time.

---

### **Results**
#### **Training Performance**:
| Metric       | Value       |
|--------------|-------------|
| Training Accuracy | 99.63% |
| Validation Accuracy | 98.72% |

#### **Testing Performance**:
| Metric       | Value       |
|--------------|-------------|
| Test Accuracy | ~98%       |
| F1-score      | ~97%       |

The model demonstrates high accuracy and robustness across all classes.

---

### **Streamlit App**
The app provides an interactive interface for real-time traffic sign recognition.

#### Key Features:
1. Image Upload: Users can upload `.jpg`, `.jpeg`, or `.png` files.
2. Real-Time Prediction: The app predicts the class label with high confidence.
3. Confidence Scores: Displays confidence scores for all classes (optional).
4. Deployment URL: [Streamlit App URL](https://traffic-sign-recognition-using-mobilenetv2.streamlit.app/)

---

### **Installation & Usage**
#### Local Setup:
1. Clone the repository:
   ```bash
   git clone https://github.com/nainesh-20/traffic_sign_recognition.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app locally:
   ```bash
   streamlit run app.py
   ```

#### Online Access:
Visit the deployed app at [Streamlit App URL](https://traffic-sign-recognition-using-mobilenetv2.streamlit.app/).

---

### **Future Improvements**
1. Enhance accuracy by experimenting with other pre-trained models like EfficientNet or ResNet.
2. Add support for batch image uploads.
3. Integrate live webcam inference for real-time detection.
4. Deploy on platforms like AWS or Google Cloud for scalability.

---

### **References**
1. [German Traffic Sign Recognition Benchmark Dataset](https://gts.ai/case-study/german-traffic-sign-recognition-dataset/)
2. [MobileNet v2 Paper](https://arxiv.org/pdf/1801.04381.pdf)
3. [Streamlit Documentation](https://docs.streamlit.io/)

---

### **Acknowledgments**
Special thanks to the creators of the GTSRB dataset and the PyTorch community for providing tools to build this project.

---

This README file provides a comprehensive overview of your project and serves as documentation for others who want to understand or build upon your work! Let me know if you need further edits or additions!

Citations:
[1] https://gts.ai/case-study/german-traffic-sign-recognition-dataset/
[2] https://pmc.ncbi.nlm.nih.gov/articles/PMC10900943/
[3] https://docs.ultralytics.com/guides/streamlit-live-inference/
[4] https://github.com/sovit-123/German-Traffic-Sign-Recognition-with-Deep-Learning
[5] https://www.youtube.com/watch?v=Tv8L2o7fAFc
[6] https://github.com/maxritter/SDC-Traffic-Sign-Recognition
[7] https://www.sciencedirect.com/science/article/pii/S2405844022030808
[8] https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-its.2018.5171
[9] https://arxiv.org/html/2403.08283v1
[10] https://www.youtube.com/watch?v=N8TxB43y-xM
[11] https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
[12] https://www.mdpi.com/1424-8220/24/11/3282
[13] https://discuss.streamlit.io/t/yolov5-real-time-inference-for-images-and-videos-on-streamlit/36730
[14] https://xwu136.com/project_details_german_traffic_sign_recognition.html
[15] https://www.sciencedirect.com/science/article/pii/S2046043022000557
[16] https://www.datature.io/blog/building-a-simple-inference-dashboard-with-streamlit
[17] https://benchmark.ini.rub.de
[18] https://streamlit.io
[19] https://github.com/moaaztaha/Yolo-Interface-using-Streamlit
[20] https://universe.roboflow.com/mohamed-traore-2ekkp/gtsdb---german-traffic-sign-detection-benchmark

---
