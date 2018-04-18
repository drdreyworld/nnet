package nnet

type Layer interface {
	InitDataSizes(w, h, d int) (int, int, int)

	Activate(inputs *Data) (output *Data)
	Backprop(deltas *Data) (nextDeltas *Data)

	Unserialize(cfg LayerConfig) (err error)
	Serialize() (cfg LayerConfig)

	GetOutput() *Data
}

type TrainableLayer interface {
	ResetGradients()
	GetWeightsWithGradient() (w, g *Data)
	GetBiasesWithGradient() (w, g *Data)
}