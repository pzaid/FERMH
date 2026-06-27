import torch
from PIL import Image
from torchvision import transforms
import timm
import torch.nn as nn

# Define your model class (same as in train_model.py)
class CustomSwinTransformer(nn.Module):
    def __init__(self, pretrained=False, num_classes=7):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224', pretrained=pretrained, num_classes=0
        )
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CustomSwinTransformer(pretrained=False, num_classes=7)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess image
img_path = "test_face.jpg"  # Change to your image filename
img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1).item()

# Class labels (update if needed)
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
print(f"Predicted emotion: {class_names[pred]}")