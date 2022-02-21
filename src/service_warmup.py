from inc.config.utils import device, transformer, stats 
from inc.models.mobilenet import COVID_CT_MobileNet
from inc.models.resnet import COVID_CT_RESNET
import torch

def init():
    settings = {
        'DEVICE': device,
        'TRANSFORM': transformer,
        'STATS': stats,
        'RESNET_PATH': '../models/covid_ct_resnet_checkpoint.pth',
        'MOBILE_PATH': '../models/covid_ct_mobile_checkpoint.pth',
        'RESULTS': '../images/results.png',
    }

    resnet = COVID_CT_RESNET(3).to(device)
    resnet.eval()
    resnet.load_state_dict(torch.load(settings['RESNET_PATH'], map_location=settings['DEVICE']))

    mobilenet = COVID_CT_MobileNet(3).to(device)
    mobilenet.eval()
    mobilenet.load_state_dict(torch.load(settings['MOBILE_PATH'], map_location=settings['DEVICE']))
    
    settings['resnet'] = resnet
    settings['mobilenet'] = mobilenet

    return settings