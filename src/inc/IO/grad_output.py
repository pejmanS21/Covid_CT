from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch
from ..config.utils import stats, classes


def denormalizer(image, stats=stats):
    if len(image.size()) == 4:
        image = image[0]
    image = image.cpu().numpy().transpose(1, 2, 0)
    mean, std = stats
    image = image * std + mean
    image = np.clip(image, 0, 1)

    return image


def visualizer(image, model, model_type: str = 'resnet', result_path: str = '../images/res.png'):
    if model_type == 'resnet':
        target_layer = model.network.layer4[2].bn2
    elif model_type == 'mobilenet':
        target_layer = model.network.features[-1][-1]
    else:
        print('Check input Model!')
        return 0

    if len(image.size()) != 4:
        image = image.unsqueeze(0)

    out = model(image)
    _, pred = torch.max(out, dim=1)

    gradcam = GradCAM(model, target_layer)
    gradcam_pp = GradCAMpp(model, target_layer)

    mask, _ = gradcam(image)
    heatmap, result = visualize_cam(mask, image)
    mask_pp, _ = gradcam_pp(image)
    heatmap_pp, result_pp = visualize_cam(mask_pp, image)
    
    save_fig(image[0], result, result_pp, result_path)

    # fig, axs = plt.subplots(1, 3)
    # axs[0].imshow(denormalizer(image[0]))
    # axs[1].imshow(result.permute(1, 2, 0))
    # axs[2].imshow(result_pp.permute(1, 2, 0))

    # axs[0].axis('off')
    # axs[1].axis('off')
    # axs[2].axis('off')
    
    # fig.suptitle(classes[pred[0]])
    # plt.savefig(result_path)
    
    return pred[0], classes[pred[0]]

def save_fig(image, grad_image, grad_image_pp, results_path):
    final_image = torch.cat((torch.from_numpy(denormalizer(image).transpose(2, 0, 1)), grad_image, grad_image_pp), 2)
    save_image(final_image, results_path)

