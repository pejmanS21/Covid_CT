from inc.config.utils import device, transformer, stats 
from inc.models.mobilenet import COVID_CT_MobileNet
from inc.models.resnet import COVID_CT_RESNET

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
    resnet._load_model_(settings['RESNET_PATH'], settings['DEVICE'])
    mobilenet = COVID_CT_MobileNet(3).to(device)
    mobilenet._load_model_(settings['MOBILE_PATH'], settings['DEVICE'])
    
    settings['resnet'] = resnet
    settings['mobilenet'] = mobilenet

    return settings