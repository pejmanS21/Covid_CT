import torch
import torchvision.transforms as T


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ['NonCOVID', 'COVID', 'CAP']
# stats = (0.5,), (0.5,)
stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
size = 224
batch_size=128
channels = 3

transformer = T.Compose([
    T.Resize(size), 
    T.CenterCrop(size),
    T.ToTensor(),
    T.Normalize(*stats)
])