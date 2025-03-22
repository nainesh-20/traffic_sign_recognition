import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2

# Load pre-trained MobileNet v2 and modify classifier layer
model = mobilenet_v2(weights=None)  # No pre-trained weights since we're loading fine-tuned weights
model.classifier[1] = torch.nn.Linear(model.last_channel, 43)  # Replace with number of classes

# Load fine-tuned weights
model.load_state_dict(torch.load("mobilenet_v2_traffic_signs.pth", map_location=torch.device('cpu')))
model.eval()


# Load the trained model weights
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    model = mobilenet_v2(weights=None)  # No pre-trained weights
    model.classifier[1] = torch.nn.Linear(model.last_channel, 43)  # Replace with number of classes
    model.load_state_dict(torch.load("mobilenet_v2_traffic_signs.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()


# Define preprocessing transforms (must match training preprocessing)
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to match input size
    transforms.ToTensor(),       # Convert image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize using training mean/std
])

# Class labels (replace with actual labels from your dataset)
class_labels = [
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

# Streamlit app layout
st.title("Traffic Sign Recognition")
st.write("Upload an image of a traffic sign and let the model predict its class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted_class = torch.max(outputs, 1)

        # Display prediction
        predicted_label = class_labels[predicted_class.item()]
        st.write(f"**Predicted Class:** {predicted_label}")
    
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
