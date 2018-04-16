package nnet

type LossFunction func(target, result *Data) float64

var LossRegistry = lossRegistry{}

type lossRegistry map[string]LossFunction
