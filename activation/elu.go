package nn

import (
	"math"
	"math/rand"
	"github.com/drdreyworld/nnet"
)

const ACTIVATION_ELU = "elu"

func init() {
	nnet.ActivationsRegistry[ACTIVATION_ELU] = &ActivationELU{}
}

type ActivationELU struct{}

func (a *ActivationELU) Forward(v float64) float64 {
	if v < 0 {
		v = 0.5 * (math.Exp(v))
	}
	return v
}

func (a *ActivationELU) Backward(v float64) float64 {
	if v < 0 {
		return rand.Float64() / 100
	}
	return 1
}

func (a *ActivationELU) Serialize() string {
	return ACTIVATION_ELU
}
