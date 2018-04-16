package nnet

type Trainer interface {
	SetNet(n NNet)
	Activate(inputs, target *Data) *Data
	UpdateWeights()
}
