package activation

import (
	"github.com/drdreyworld/nnet"
	"math"
)

const ACTIVATION_ELU = "elu"

func init() {
	nnet.ActivationsRegistry[ACTIVATION_ELU] = func(params interface{}) nnet.Activation {
		r := &ActivationELU{}
		r.SetParams(params)
		return r
	}
}

type ActivationELU struct {
	K float64
}

func (a *ActivationELU) SetParams(i interface{}) {
	a.K = 0.01

	if p, ok := i.(map[string]interface{}); ok {
		if k, ok := p["K"].(float64); ok {
			a.K = k
		}
	}
}

func (a *ActivationELU) Forward(v float64) float64 {
	if v <= 0 {
		return a.K * (math.Exp(v) - 1)
	}
	return v
}

func (a *ActivationELU) Backward(v float64) float64 {
	if v <= 0 {
		return a.K * math.Exp(v)
	}
	return 1
}
