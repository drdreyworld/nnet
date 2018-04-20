package nnet

type Net interface {
	Init() (err error)
	Activate(inputs *Data) (output *Data)
	Backprop(deltas *Data) (gradient *Data)
	GetOutputDeltas(target, output *Data) (res *Data)
	GetLayersCount() int
	GetLayer(index int) Layer
	GetLoss(target, output *Data) float64
}