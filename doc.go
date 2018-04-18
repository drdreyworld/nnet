/*
Package nnet defines base primitives and interfaces for neural network operations.

Now it can be used for construction feed forward networks (FC and CNN).

Realized basic layer types:

	- dense or fully connected layer
	- convolution layer
	- pooling layer
	- activation layer (with target activation function)
	- softmax layer (as classifier)

It is possible to create own:

	- layer type
	- activation function
	- loss function
	- trainer type
	- storage type
 */
package nnet
