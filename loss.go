package nnet

// LossRegistry - collection of loss functions by code.
// Each new loss function must be registered before use.
var LossRegistry = map[string]LossFunction{}

type LossFunction func(target, result *Data) float64