# -*- coding: utf-8 -*-
from typing import Any, Dict, Tuple

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_tensor, to_pil_image

from styletransfer.model import FeatureDict, preprocess, StyleContentModel


class StyleTransfer:
    """Implements the Neural Style Transfer algorithm.

    An object of this class represents a reifeid run of the NST algorithm, as 
    proposed by [Gatys et al., 2016]. For a given _content_ and _style_ image, 
    the algorithm generates a new image that depicts the scene of the content 
    image, rendered in the artistic style (i.e. the texture) of the style image. 
    
    Both style and content image can be of arbitrary size, however larger sizes
    will result in slower generation. The shape of the output image always
    matches that of the content image. 

    Generation is performed via an optimization procedure that alters the 
    generated image (in pixel space) in order to minimize the L2-error between
    style/content features of the generated image and the style/content
    image, respectively. The relative contribution of each error term is 
    negotiated via the `alpha` and `beta` parameters.
    
    An additional `hot_start` flag determines how the generated image is
    initialized. If set to `True`, the optimization procedure starts with a 
    copy of the content image; this can lead to faster results at the cost of 
    biasing the generated image towards the original. Otherwise, each pixel of 
    the generated image is seeded independently from a standard normal 
    distribution. 

    Args:
        model: The model from which style and content features are computed.
        content_image: The content image (see above).
        style_image: The style image (see above).
        alpha: The weighting of the content loss, as defined in the paper.
        beta: The weighting of the style loss, as defined in the paper.
        hot_start: If True, the generated image is initialized as a copy of the
            content image, otherwise by sampling each pixel from a standard
            normal distribution.
        optim_kwargs: key-word arguments for the optimizer (torch.optim.Adam).
    """


    def __init__(self, 
        model: StyleContentModel, 
        content_image: torch.Tensor, 
        style_image: torch.Tensor, 
        alpha: float,
        beta: float,
        hot_start: bool = False, 
        optim_kwargs: Dict[str, Any] = {}) -> None:

        self.model = model
        device = next(model.parameters())[0].device
        
        self.target_content, _ = self.model(preprocess(content_image).to(device))
        _, self.target_style = self.model(preprocess(style_image).to(device))

        if hot_start:
            self._image = content_image.clone()
            self._image.requires_grad_(True)
        
        else:
            self._image = torch.randn_like(
                content_image, 
                device=device, 
                requires_grad=True
            )

        self.optimizer = torch.optim.Adam(
            [self._image], 
            **optim_kwargs
        )
        
        self.alpha = alpha
        self.beta = beta


    def step(self) -> Tuple[float, float, float]:
        """Performs one step of optimization on the generated image.

        Calling this method performs one step of optimization on the generated
        image. This is usually not sufficent for the generated image to
        converge, however it can be helpful for hyperparameter selection.
        
        Returns:
            The total loss, the (scaled) content loss, and the (scaled) style
            loss.
        """
        
        self.optimizer.zero_grad()

        image_content, image_style = self.model(preprocess(self._image))

        content_loss = self.alpha * self._loss(image_content, self.target_content)
        style_loss = self.beta * self._loss(image_style, self.target_style)

        loss = content_loss + style_loss

        loss.backward()
        self.optimizer.step()

        return loss.item(), content_loss.item(), style_loss.item()


    def _loss(
        self, 
        inputs: FeatureDict, 
        targets: FeatureDict) -> torch.Tensor:
        """Computes the MSE between target and reconstruction features"""

        layers = inputs.keys()

        loss = sum(F.mse_loss(inputs[l], targets[l]) for l in layers)
        loss /= len(layers)
        
        return loss


    @property
    def image(self) -> Image:
        """The generated image.
        
        Modifying this image does not impact the optimization procedure.
        """
        return to_pil_image(self._image.detach().cpu().squeeze(0))