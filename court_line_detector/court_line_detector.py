import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

class  CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        

teste = CourtLineDetector('models/last.pt')
# print(teste.model)
print(type(teste.model))    
# print(type(teste.transform))
# print(type(teste.model.fc))