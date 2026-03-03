import torch
import cv2
from PIL import Image
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def classify_roi(model, roi, class_names):
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_pil = Image.fromarray(roi_rgb)

    tensor = transform(roi_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return class_names[pred.item()], confidence.item()