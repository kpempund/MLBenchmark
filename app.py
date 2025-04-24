import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# load the model
checkpoint = torch.load('model.pth', map_location='cpu')
model_type = checkpoint['model_type']
num_classes = checkpoint['num_classes']
model_state_dict = checkpoint['model_state_dict']

# set up data transformation
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if model_type == 'inception_v3':
        transform = T.Compose([
            T.Resize(299),
            T.CenterCrop(299),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
else:
        modelname = {
        'efficientnet_b0': 'EfficientNet',
        'resnet50': 'ResNet50',
        'vit_b_16': 'ViT',
        'squeezenet1_0': 'SqueezeNet',
        'inception_v3': 'Inception-v3',
        'mobilenet_v3_large': 'MobileNet-v3'
        }

# check the model type and make the model
if model_type == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
elif model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
elif model_type == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=False)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
elif model_type == 'vit_b_16':
        model = models.vit_b_16(pretrained=False)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
elif model_type == 'squeezenet1_0':
        model = models.squeezenet1_0(pretrained=False)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
elif model_type == 'inception_v3':
        model = models.inception_v3(pretrained=False, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
else:
        raise ValueError(f"Unsupported model type: {model_type}")

# load state dict(save weight) and eval(eval mode)
model.load_state_dict(model_state_dict)
model.eval()

# set classes
classes = ['ants', 'bees']

st.title("Ants vs. Bees Image Classifier")

st.write("Model type: ", modelname.get(model_type))

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
uploaded_files = st.file_uploader("Upload all images from a folder...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def classify_image(file):
    image = Image.open(file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_index = torch.argmax(probabilities).item()
        predicted_class = classes[predicted_class_index]
        confidence = probabilities[predicted_class_index].item() * 100

    return {
        "filename": file.name,
        "image": image,
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }

if 'results' not in st.session_state:
    st.session_state.results = []
if 'index' not in st.session_state:
    st.session_state.index = 0

if uploaded_files and st.button("Classify All Images"):
    with st.spinner("Classifying images..."):
        with ThreadPoolExecutor(max_workers=8) as executor:
            st.session_state.results = list(executor.map(classify_image, uploaded_files))
        st.session_state.index = 0
        st.success("Classification complete!")

results = st.session_state.get("results", [])
if results:
    current = st.session_state.index
    total = len(results)
    result = results[current]

    st.subheader(f"Image {current + 1} of {total}: {result['filename']}")
    st.image(result['image'], caption=f"Prediction: {result['prediction']} ({result['confidence']:.2f}%)", use_container_width=True)
    st.subheader("Class Probabilities")
    for i, class_name in enumerate(classes):
        st.write(f"{class_name}: {result['probabilities'][i].item()*100:.2f}%")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Previous", disabled=current <= 0):
            st.session_state.index -= 1
            st.rerun()
    with col3:
        if st.button("Next ➡️", disabled=current >= total - 1):
            st.session_state.index += 1
            st.rerun()