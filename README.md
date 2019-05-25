# nnet

[![GoDoc](https://godoc.org/github.com/drdreyworld/nnet?status.svg)](https://godoc.org/github.com/drdreyworld/nnet)

version: 0.0.1

Extendable golang neural network library.

Can be used for construction feed forward networks.

Layer types:

	- FC - multidimentional fully connected layer
	- Conv - convolution layer
	- Pool - max pooling layer (downsampling)
	- Activation - activation layer (with target activation function and specific params for it)
	- Softmax - multidimentional layer for classification

Activation functions:

    - Sigmoid
    - Tanh
    - ReLU
    - ELU
    - LeakyReLU
    - Softplus

Created and trained network can be saved to (loaded from) json-file.

Library can be extended by creating your own:

	- layer type
	- activation function
	- loss function
	- trainer type
	- storage type