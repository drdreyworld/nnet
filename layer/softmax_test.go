package layer

import (
	"testing"
	"github.com/drdreyworld/nnet"
)

func TestSoftmax_Complex(t *testing.T) {
	cfg := LayerConfigSoftmax()

	l, err := LayerConstructorSoftmax(cfg)
	if err != nil {
		t.Errorf("create layer error:", err.Error())
	}

	iw, ih, id := 3, 1, 1
	ow, oh, od := l.InitDataSizes(iw, ih, id)

	if iw != ow || ih != oh || id != od {
		t.Error("output sizes not equal input sizes")
	}

	inputs := &nnet.Data{}
	inputs.InitVector(iw)
	inputs.Data = []float64{0.1, 0.6, 0.3}

	target := inputs.CopyZero()
	target.Data = []float64{
		0.25838965173797984,
		0.42601251494920570,
		0.31559783331281430,
	}

	output := l.Activate(inputs)

	if l.GetOutput() != output {
		t.Error("getOutput returns other result than activate")
	}

	if output.Dims[0] != ow || output.Dims[1] != oh || output.Dims[2] != od {
		t.Error("output dimentions mistmatch")
	}

	for i := 0; i < iw; i++ {
		if output.Data[i] != target.Data[i] {
			t.Error("invalid output value")
		}
	}

	deltas := &nnet.Data{}
	deltas.InitVector(iw)
	deltas.FillRandom(-1, 1)

	// backprop in softmax not change deltas because
	// it's last layer and gradient must be equal network deltas
	gradient := l.Backprop(deltas)

	for i := 0; i < iw; i++ {
		if gradient.Data[i] != deltas.Data[i] {
			t.Error("gradient value is changed")
		}
	}

	config := l.Serialize()

	if config.Type != LAYER_SOFTMAX {
		t.Error("invalid layer type in serialized config")
	}
}