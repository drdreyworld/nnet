package activation

import (
	"github.com/drdreyworld/nnet"
	"math"
)

const ACTIVATION_TANH = "tanh"

func init() {
	nnet.ActivationsRegistry[ACTIVATION_TANH] = func(params interface{}) nnet.Activation {
		r := &ActivationTanh{}
		r.SetParams(params)
		return r
	}
}

type ActivationTanh struct {
	K float64
}

func (a *ActivationTanh) SetParams(interface{}) {}

func (a *ActivationTanh) Forward(v float64) float64 {
	eZ := math.Exp(v)
	ez := math.Exp(-v)

	return (eZ - ez) / (eZ + ez)
}

func (a *ActivationTanh) Backward(v float64) float64 {
	return 1 - v*v
}
