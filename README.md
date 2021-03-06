## Implementation of "Rethinking Conditional GAN training: an approach using geometrically structured latent manifolds." [[PDF](https://arxiv.org/abs/2011.13055)]

![alt text](./images/im1.jpg "Title")

# Requirements
Tensorflow = 1.9
Python = 3.x
Scipy = 1.4

# Data preparation
The training files should be placed at **data_dir/images/training/**

The validation files should be placed at **data_dir/images/validation/**

Each training and validation file should be a concatination of the input and the corresponding output. An example is shown below:

![alt text](./images/30_AB.jpg "Title")

# Train
> python p2p_geo.py --mode=train --data_dir=*data directory* --learning_rate=1-e4 --logs_dir=*logs directory* --batch_size=25

# Visualize
> python p2pGeo.py --mode=visualize


