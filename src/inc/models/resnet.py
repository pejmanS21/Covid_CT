import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary


class COVID_CT_RESNET(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=True)
        # Replace last layer
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)
    
    def _summary_(self, input_size=(3, 224, 224)):
        return summary(self, input_size)

    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = COVID_CT_RESNET(num_classes=3).to(device)
    # model._load_model_('../../../models/covid_ct_resnet_checkpoint.pth', device)
    model._summary_((3, 224, 224))
