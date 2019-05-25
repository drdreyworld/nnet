package activation

import (
	"github.com/drdreyworld/nnet"
	"math"
)

const ACTIVATION_SIGMOID = "sigmoid"

func init() {
	nnet.ActivationsRegistry[ACTIVATION_SIGMOID] = func(params interface{}) nnet.Activation {
		r := &ActivationSigmoid{}
		r.SetParams(params)
		return r
	}
}

type ActivationSigmoid struct{}

func (a *ActivationSigmoid) SetParams(interface{}) {
}

func (a *ActivationSigmoid) Forward(v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

func (a *ActivationSigmoid) Backward(v float64) float64 {
	return v * (1 - v)
}
