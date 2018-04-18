package layer

import (
	"github.com/drdreyworld/nnet"
	"testing"
)

func TestDense_Complex(t *testing.T) {
	cfg := LayerConfigDense(1, 2, 3)

	l, err := LayerConstructorDense(cfg)
	if err != nil {
		t.Errorf("create layer error:", err.Error())
	}

	iw, ih, id := 2, 2, 2
	ow, oh, od := l.InitDataSizes(iw, ih, id)

	if ow != 1 || oh != 2 || od != 3 {
		t.Error("output sizes not equal input sizes")
	}

	inputs := &nnet.Data{}
	inputs.InitCube(iw, ih, id)
	inputs.Data = []float64{
		10, 11,
		12, 13,

		14, 15,
		16, 17,
	}

	output := l.Activate(inputs)

	if l.GetOutput() != output {
		t.Error("getOutput returns other result than activate")
	}

	if output.Dims[0] != ow || output.Dims[1] != oh || output.Dims[2] != od {
		t.Error("output dimentions mistmatch")
	}

	config := l.Serialize()

	if config.Type != LAYER_DENSE {
		t.Error("invalid layer type in serialized config")
	}

	if config.Data.Int("OWidth") != 1 {
		t.Error("missed output width in serialized config")
	}

	if config.Data.Int("OHeight") != 2 {
		t.Error("missed output height in serialized config")
	}

	if config.Data.Int("ODepth") != 3 {
		t.Error("missed output depth in serialized config")
	}

	if weights, ok := config.Data["Weights"].(nnet.Data); !ok {
		t.Error("missed weights in serialized config")
	} else if len(weights.Data) != iw*ih*id*ow*oh*od {
		t.Error("invalid weights length in serialized config")
	}

	if biases, ok := config.Data["Biases"].(nnet.Data); !ok {
		t.Error("missed biases in serialized config")
	} else if len(biases.Data) != ow*oh*od {
		t.Error("invalid biases length in serialized config")
	}

	// @todo test calc output algorithm
}
