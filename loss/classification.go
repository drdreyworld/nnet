package loss

import (
	"math"
	"github.com/drdreyworld/nnet"
)

const LOSS_CLASSIFICATION = "classification"

func init() {
	nnet.LossRegistry[LOSS_CLASSIFICATION] = Classification
}

func Classification(target, result *nnet.Data) (res float64) {
	for i := 0; i < len(target.Data); i++ {
		if target.Data[i] == 1 {
			return -math.Log(result.Data[i])
		}
	}
	return 0
}

