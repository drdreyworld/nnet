package nnet

type Trainer interface {
	Train(inputs, target Data)
}
