package nnet

type Layer interface {
	Init(cfg LayerConfig) (err error)
	InitDataSizes(w, h, d int) (int, int, int)
	Activate(inputs *Data) (output *Data)
	Backprop(deltas *Data) (nextDeltas *Data)
	Serialize() LayerConfig
	GetOutput() *Data
}

type TrainableLayer interface {
	ResetGradients()
	GetWeightsWithGradient() (w, g *Data)
	GetBiasesWithGradient() (w, g *Data)
}