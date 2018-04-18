package nnet

type Trainer interface {
	SetNet(n Net)
	Activate(inputs, target *Data) *Data
	UpdateWeights()
}
