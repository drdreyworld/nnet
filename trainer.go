package nnet

type Trainer interface {
	Train(inputs, target Mem)
}
