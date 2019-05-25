package activation

import "github.com/drdreyworld/nnet"

const ACTIVATION_RELU = "relu"

func init() {
	nnet.ActivationsRegistry[ACTIVATION_RELU] = func(params interface{}) nnet.Activation {
		r := &ActivationReLU{}
		r.SetParams(params)
		return r
	}
}

type ActivationReLU struct{}

func (a *ActivationReLU) SetParams(interface{}) {
}

func (a *ActivationReLU) Forward(v float64) float64 {
	if v < 0 {
		v = 0
	}
	return v
}

func (a *ActivationReLU) Backward(v float64) float64 {
	if v <= 0 {
		return 0
	}
	return 1
}
