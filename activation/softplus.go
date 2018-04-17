package nn

import (
	"math"
	"github.com/drdreyworld/nnet"
)

const ACTIVATION_SOFTPLUS = "softplus"

func init() {
	nnet.ActivationsRegistry[ACTIVATION_SOFTPLUS] = &ActivationSoftPlus{}
}

type ActivationSoftPlus struct{}

func (a *ActivationSoftPlus) Forward(v float64) float64 {
	return math.Log10(1 / (1 + math.Exp(v)))
}

func (a *ActivationSoftPlus) Backward(v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

func (a *ActivationSoftPlus) Serialize() string {
	return ACTIVATION_SOFTPLUS
}
