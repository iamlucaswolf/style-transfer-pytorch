# -*- coding: utf-8 -*-
from typing import List, Tuple, Set, Dict

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms.functional import normalize


LayerIndex = Tuple[int, int]
"""Typedef for indexing VGG19 layers.

A LayerIndex (block, layer) specifies a convolutional layer in a particular 
block in the 19-layer variant of the VGG network (see 
[Symonian & Zisserman, 2015] for details). Indices start at one; pooling layers
are skipped implicitly. 

Example:
    (3, 4) # Specifies the fourth conv layer in the third block
    (4, 1) # Specifies the first conv layer in the fourth block
    # Note that the interleaved Max-Pool layer has been skipped implicitly
"""


FeatureDict = Dict[LayerIndex, torch.Tensor]
"""Typedef for features extracted by the model.

A FeatureDict maps VGG-19 layers (specifed by their corresponding LayerIndex) to
the respective features extracted from that layer.
"""


# Mean and standard deviation of the ImageNet dataset, used for standardization
_imagenet_mean = [0.485, 0.456, 0.406]
_imagenet_std = [0.229, 0.224, 0.225]


def preprocess(image):
    """Preprocess images for use with StyleContentModel

    Clips pixel values to [0, 1] and standardizes the resulting image w.r.t. the
    training set distribution of ImageNet.
    """

    # Restrict pixel values to [0, 1]. This is necessary, since standardization
    # expects inputs from this range, but images produced by NST do not 
    # always satisfy this assumption.
    image.data.clamp_(0, 1)
    
    # Standardize input
    mean = torch.tensor(_imagenet_mean, device=image.device).view(-1, 1, 1)
    std = torch.tensor(_imagenet_std, device=image.device).view(-1, 1, 1)

    image = (image - mean) / std
    return image
 

class StyleContentModel(nn.Module):
    """Models the content and style of an image.

    This model extracts content and style information contained in the input 
    image using the feature spaces proposed by [Gatys et al., 2015]. The
    underlying network is the 19-layer VGG implementation from the torchvision
    package.
    
    A forward pass of the model returns two `FeatureDict`s, containing the 
    layer activations (content) and gram matrices (style) produced by the model.
    The layers from which the respective features are computed can be selected 
    via constructor arguments. 

    Args:
        content_layers: layers from which content information is computed
        style_layers: layers from which style information is computed

    Example:
        >>> model = StyleContentModel(
        >>>     content_layers=[(3, 2)], 
        >>>     style_layers=[(1, 1), (2, 1), (3, 1)]
        >>> )
        >>> content, style = model(preprocess(image))

    Note: This is not a "classical" nn.Module, in the sense that it does not
        contain any parameters tunable by gradient descent. However, using the 
        Module interface is practical, as it allows for easily moving the model
        to GPU (e.g. via `model.to('cuda')`).

    Note: This model uses pre-trained from torchvision's VGG-19 implementation.
        When StyleContentModel is instantiated for the first time, this may 
        cause torchvision to download the model weights, which may take some
        time, depending on your internet connection.
    """

    def __init__(
        self, 
        content_layers: List[LayerIndex], 
        style_layers: List[LayerIndex]) -> None:

        super(StyleContentModel, self).__init__()

        # List of convolutional layers that need to be evaluated (in this order)
        # to produce the necessary activations
        self._layers : List[LayerIndex] = []
        
        # Convolutional layers from which content/style features are computed
        self.content_layers: Set[LayerIndex] = set(content_layers)
        self.style_layers: Set[LayerIndex] = set(style_layers)

        vgg = models.vgg19(pretrained=True)

        # The last layer that needs to be considered to compute all necessary
        # activations
        max_layer = max(max(content_layers), max(style_layers))

        block_idx = 1
        layer_idx = 1

        for module in vgg.features: 
            
            if isinstance(module, nn.Conv2d):

                conv = copy.deepcopy(module)

                # Disable backpropagation for copied layers
                conv.weight.requires_grad = False
                conv.bias.requires_grad = False

                # Add layer to the network
                setattr(self, f'conv{block_idx}_{layer_idx}', conv)
                
                layer = (block_idx, layer_idx)
                self._layers.append(layer)

                if layer == max_layer:
                    break

                layer_idx += 1
            
            # A MaxPool layer marks the end of a block in VGG
            if isinstance(module, nn.MaxPool2d):
                block_idx += 1
                layer_idx = 1

   
    def forward(self, x: torch.Tensor) -> Tuple[FeatureDict, FeatureDict]:

        # The computed content/style features
        content_features: Dict[str, torch.Tensor] = {}
        style_features: Dict[str, torch.Tensor] = {}

        block = 1

        for layer in self._layers:
            block_idx, layer_idx = layer

            # Interleave pooling layer if the current layer is in a new block
            if block_idx > block:
                
                # Gatys et al. suggest using average pooling instead of the max
                # pooling layers used in VGG for improved visual results on NST
                x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
                block += 1
            
            x = F.relu(getattr(self, f'conv{block_idx}_{layer_idx}')(x))

            # If the current layer contains activations from which style/content
            # features are to be computed, save the results to the respective
            # FeatureDict
            if layer in self.content_layers:
                content_features[layer] = x

            if layer in self.style_layers:
                style_features[layer] = self.extract_style(x)

        return content_features, style_features


    def extract_style(self, activations: torch.Tensor) -> torch.Tensor:
        """Computes style information from a given set of layer activations."""
        gram_matrix = torch.einsum('nchw,ndhw->ncd', activations, activations)
        return gram_matrix


