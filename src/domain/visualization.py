import math

import numpy as np
from pytorch_grad_cam import (AblationCAM, EigenCAM, FullGrad, GradCAM,
                              GradCAMPlusPlus, HiResCAM, ScoreCAM, XGradCAM)
from pytorch_grad_cam.utils.image import show_cam_on_image


# Custom reshape transform for Siglip model
def siglip_reshape_transform(tensor, height=None, width=None):
    """
    Siglip models don't use a CLS token, so we reshape all patches
    """

    batch_size = tensor.size(0)
    num_tokens = tensor.size(1)
    channels = tensor.size(2)

    # Calculate height and width from number of tokens
    h = w = int(math.sqrt(num_tokens))

    # Reshape: (batch, num_patches, channels) -> (batch, height, width, channels)
    result = tensor.reshape(batch_size, h, w, channels)

    # Transpose to (batch, channels, height, width) for CNN format
    result = result.permute(0, 3, 1, 2)
    return result


def get_visualize_gradcam(
    model, target_layers, targets, input_tensor, image, image_size
):
    # Construct the CAM object once, and then re-use it on many images.
    # Use custom reshape_transform for Siglip models
    with EigenCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=siglip_reshape_transform,
    ) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]

        # Convert image to numpy array and normalize to [0, 1]
        image_np = np.array(image.resize((image_size, image_size))) / 255.0

        visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
        # # You can also get the model outputs without having to redo inference
        # model_outputs = cam.outputs

    return visualization
