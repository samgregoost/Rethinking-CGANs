## Implementation of "Rethinking Conditional GAN training: an approach using geometrically structured latent manifolds." [[PDF](https://arxiv.org/abs/2011.13055)]

![alt text](./images/im1.jpg "Title")

# Requirements
Tensorflow = 1.9
Python = 3.x
Scipy = 1.4

# Data preparation
The training files should be put in data_dir/images/training/
The validation files should be put in data_dir/images/validation/

Each training and validation file should be a concatination of the input and the corresponding output. 

# Train
python p2pGeo.py --mode=train

# Visualize
python p2pGeo.py --mode=visualize


