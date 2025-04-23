import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image

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
        # transform = T.Compose([
        #     T.Resize(299),
        #     T.CenterCrop(299),
        #     T.ToTensor(),
        #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
else:
        raise ValueError(f"Unsupported model type: {model_type}")

# load state dict(save weight) and eval(eval mode)
model.load_state_dict(model_state_dict)
model.eval()

# set classes
classes = ['ants', 'bees']

st.title("Ants vs. Bees Image Classifier")

st.write("Model type: ", modelname.get(model_type))

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    if st.button("Classify"):
        with st.spinner("Classifying..."):
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0) # Softmax for probabilities
                predicted_class_index = torch.argmax(probabilities).item()
                predicted_class = classes[predicted_class_index]
                confidence = probabilities[predicted_class_index].item() * 100

            st.header("Prediction")
            st.write(f"The image is a {predicted_class} with {confidence:.2f}% confidence.")

            # Display probabilities for each class
            st.subheader("Class Probabilities")
            for i, class_name in enumerate(classes):
              st.write(f"{class_name}: {probabilities[i].item()*100:.2f}%")