package activation

import (
	"github.com/drdreyworld/nnet"
)

// LeakyReLU
const ACTIVATION_LRELU = "LRELU"

func init() {
	nnet.ActivationsRegistry[ACTIVATION_LRELU] = func(params interface{}) nnet.Activation {
		r := &ActivationLRELU{}
		r.SetParams(params)
		return r
	}
}

type ActivationLRELU struct {
	K float64
}

func (a *ActivationLRELU) SetParams(i interface{}) {
	a.K = 0.01

	if p, ok := i.(map[string]interface{}); ok {
		if k, ok := p["K"].(float64); ok {
			a.K = k
		}
	}
}

func (a *ActivationLRELU) Forward(v float64) float64 {
	if v <= 0 {
		return -a.K * v
	}
	return v
}

func (a *ActivationLRELU) Backward(v float64) float64 {
	if v <= 0 {
		return a.K
	}
	return 1
}
