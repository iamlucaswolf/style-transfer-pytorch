# Neural Style Transfer

A PyTorch implementation of 
[Gatys et al. 2015](https://arxiv.org/pdf/1508.06576.pdf).

## TODOs

- rename preprocess -> load_image
- preprocess inputs to StyleContentModel -> unsqueeze(0), normalize
- clip reconstruction before input