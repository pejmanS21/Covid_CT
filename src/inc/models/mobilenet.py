import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary


class COVID_CT_MobileNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Use a pretrained model
        self.network = models.mobilenet_v2(pretrained=True)
        # Replace last layer
        self.network.classifier[1] = nn.Linear(self.network.classifier[1].in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)
    
    def _summary_(self, input_size=(3, 224, 224)):
        return summary(self, input_size)
    
    def _load_model_(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = COVID_CT_MobileNet(num_classes=3).to(device)
    model._load_model_('../../../models/covid_ct_mobile_checkpoint.pth', device)
    model._summary_((3, 224, 224))
