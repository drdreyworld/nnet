package nnet_test

import (
	"encoding/json"
	"fmt"
	"github.com/drdreyworld/nnet"
	"github.com/drdreyworld/nnet/activation"
	"github.com/drdreyworld/nnet/layer"
	"github.com/drdreyworld/nnet/loss"
	"testing"
)

func TestNet_Json(t *testing.T) {
	n := &nnet.NetDefault{
		IWidth:  2,
		IHeight: 1,
		IDepth:  1,

		OWidth:  1,
		OHeight: 1,
		ODepth:  1,

		Loss: loss.LOSS_REGRESSION,

		Layers: nnet.Layers{
			&layer.Dense{
				IWidth:  2,
				IHeight: 1,
				IDepth:  1,

				OWidth:  5,
				OHeight: 5,
				ODepth:  5,
			},

			&layer.Activation{
				ActFunc: activation.ACTIVATION_SIGMOID,
			},

			&layer.Conv{
				FWidth:  3,
				FHeight: 3,
				FCount:  16,
			},

			&layer.Softmax{},
		},
	}

	n.Init()

	b, err := json.Marshal(n)
	if err != nil {
		t.Error("json matshal error:", err.Error())
	}

	nn := &nnet.NetDefault{}

	if err := json.Unmarshal(b, nn); err != nil {
		fmt.Println("json unmarshal error:", err.Error())
	}
}
