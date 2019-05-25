package loss

import (
	"github.com/drdreyworld/nnet"
	"math"
)

const LOSS_REGRESSION = "regression"

func init() {
	nnet.LossRegistry[LOSS_REGRESSION] = Regression
}

func Regression(target, result *nnet.Data) (res float64) {
	for i := 0; i < len(target.Data); i++ {
		res += math.Pow(float64(result.Data[i]-target.Data[i]), 2)
	}
	return float64(1/len(target.Data)) * res
}
