import torch
from torchvision import transforms, models
import cv2
import numpy as np

# Load pre-trained EfficientNet and modify the classifier for binary output
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)  # Change to binary output
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_frame(frame):
    """Predict if a frame is real or fake."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(frame).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()  # Apply sigmoid for binary output
    return prob

def process_video(video_path):
    """Process video and predict for each frame."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fake_probabilities = []

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        prob = predict_frame(frame)
        fake_probabilities.append(prob)

    cap.release()
    return fake_probabilities

# Test the model
video_path = "../Videos/jenifer.mp4"  # Replace with your video path
probabilities = process_video(video_path)

# Analyze results
average_fake_prob = np.mean(probabilities)
print(f"Average fake probability: {average_fake_prob:.2f}")
if average_fake_prob > 0.45:
    print("The video is likely a deepfake!")
else:
    print("The video is likely real.")
