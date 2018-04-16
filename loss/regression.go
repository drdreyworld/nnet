package loss

import (
	"math"
	"github.com/drdreyworld/nnet"
)

const LOSS_REGRESSION = "regression"

func init() {
	nnet.LossRegistry[LOSS_REGRESSION] = Regression
}

func Regression(target, result *nnet.Data) (res float64) {
	for i := 0; i < len(target.Data); i++ {
		res += math.Pow(result.Data[i]-target.Data[i], 2)
	}
	return 0.5 * res
}

