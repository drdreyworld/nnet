package nn

import (
	"math"
	"github.com/drdreyworld/nnet"
)

const ACTIVATION_SIGMOID = "sigmoid"

func init() {
	nnet.ActivationsRegistry[ACTIVATION_SIGMOID] = &ActivationSigmoid{}
}

type ActivationSigmoid struct{}

func (a *ActivationSigmoid) Forward(v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

func (a *ActivationSigmoid) Backward(v float64) float64 {
	return v * (1 - v)
}

func (a *ActivationSigmoid) Serialize() string {
	return ACTIVATION_SIGMOID
}
