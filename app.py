# ================================
# STREAMLIT WEB APP - app.py
# ================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models as torchvision_models
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
# import io
# import base64

# Page configuration
st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #FF6B6B;
    text-align: center;
    margin-bottom: 2rem;
}
.accuracy-badge {
    background-color: #4ECDC4;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
    font-size: 1.2rem;
}
.prediction-box {
    background-color: #F7F9FC;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #4ECDC4;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Traffic sign class labels
CLASS_LABELS = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", 
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)", 
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)", 
    "No passing", "No passing for vehicles over 3.5 metric tons", 
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop", 
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry", 
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right", 
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", 
    "Road work", "Traffic signals", "Pedestrians", "Children crossing", 
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing", 
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead", 
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right", 
    "Keep left", "Roundabout mandatory", "End of no passing", 
    "End of no passing by vehicles over 3.5 metric tons"
]

@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)"""
    device = torch.device('cpu')  # Use CPU for deployment
    
    # Recreate model architecture
    model = torchvision_models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 43)
    )
    
    # Load trained weights
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    return model, device

@st.cache_data
def get_preprocessing():
    """Get preprocessing pipeline (cached)"""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def predict_image(image, model, device, preprocess):
    """Make prediction on uploaded image"""
    try:
        # Preprocess image
        image_array = np.array(image)
        preprocessed = preprocess(image=image_array)
        input_tensor = preprocessed['image'].unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        return {
            'class_id': predicted_class.item(),
            'class_label': CLASS_LABELS[predicted_class.item()],
            'confidence': confidence.item(),
            'all_probabilities': probabilities.cpu().numpy()[0]
        }
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🚦 Traffic Sign Classifier</h1>', unsafe_allow_html=True)
    
    # Accuracy badge
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown('<div class="accuracy-badge">✨ 99.72% Accuracy ✨</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.info("""
        **Architecture:** MobileNetV2
        **Dataset:** GTSRB (43 classes)
        **Training:** Class-aware augmentation
        **Validation Accuracy:** 99.77%
        **Test Accuracy:** 99.72%
        """)
        
        st.header("🎯 Features")
        st.success("""
        ✅ Real-time prediction
        ✅ Confidence scoring
        ✅ Top-3 predictions
        ✅ Production-ready model
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Traffic Sign Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg', 'ppm'],
            help="Upload a clear image of a traffic sign"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Predict button
            if st.button("🔍 Classify Traffic Sign", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Load model and preprocess
                    model, device = load_model()
                    preprocess = get_preprocessing()
                    
                    # Make prediction
                    result = predict_image(image, model, device, preprocess)
                    
                    if result:
                        # Store result in session state
                        st.session_state.prediction_result = result
    
    with col2:
        st.header("🎯 Prediction Results")
        
        if hasattr(st.session_state, 'prediction_result'):
            result = st.session_state.prediction_result
            
            # Main prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h3>🚦 Predicted Sign</h3>
                <h2 style="color: #FF6B6B;">Class {result['class_id']}: {result['class_label']}</h2>
                <h3 style="color: #4ECDC4;">Confidence: {result['confidence']:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.progress(result['confidence'])
            
            # Top 3 predictions
            st.subheader("📊 Top 3 Predictions")
            top_3_indices = np.argsort(result['all_probabilities'])[-3:][::-1]
            
            for i, idx in enumerate(top_3_indices):
                prob = result['all_probabilities'][idx]
                label = CLASS_LABELS[idx]
                
                if i == 0:
                    st.success(f"🥇 **Class {idx}**: {label} ({prob:.1%})")
                elif i == 1:
                    st.info(f"🥈 **Class {idx}**: {label} ({prob:.1%})")
                else:
                    st.warning(f"🥉 **Class {idx}**: {label} ({prob:.1%})")
        else:
            st.info("👆 Upload an image above to see predictions")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🚀 Built with Streamlit | 🐳 Deployed with Docker | 🧠 Powered by PyTorch</p>
        <p>Model trained on GTSRB dataset with advanced data engineering techniques</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
