package activation

import (
	"github.com/drdreyworld/nnet"
	"math"
)

const ACTIVATION_SOFTPLUS = "softplus"

func init() {
	nnet.ActivationsRegistry[ACTIVATION_SOFTPLUS] = func(params interface{}) nnet.Activation {
		r := &ActivationSoftPlus{}
		r.SetParams(params)
		return r
	}
}

type ActivationSoftPlus struct{}

func (a *ActivationSoftPlus) SetParams(interface{}) {
}

func (a *ActivationSoftPlus) Forward(v float64) float64 {
	return math.Log10(1 / (1 + math.Exp(v)))
}

func (a *ActivationSoftPlus) Backward(v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}
