package nnet

type Layer interface {
	InitDataSizes(w, h, d int) (int, int, int)

	Activate(inputs *Data) (output *Data)
	Backprop(deltas *Data) (nextDeltas *Data)

	GetOutput() *Data
	GetType() string
}

type TrainableLayer interface {
	ResetGradients()
	GetWeightsWithGradient() (w, g *Data)
	GetBiasesWithGradient() (w, g *Data)
}