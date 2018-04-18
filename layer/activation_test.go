package layer

import (
	"testing"
	"github.com/drdreyworld/nnet"
)

type testActivationDoubleValue struct{}

func (a *testActivationDoubleValue) Forward(v float64) float64 {
	return 2 * v
}

func (a *testActivationDoubleValue) Backward(v float64) float64 {
	return 0.5 * v
}

func (a *testActivationDoubleValue) Serialize() string {
	return "testActivationDoubleValue"
}

func TestActivationConstructor(t *testing.T) {
	cfg := LayerConfigActivation(&testActivationDoubleValue{})

	l, err := LayerConstructorActivation(cfg)
	if err != nil {
		t.Errorf("create layer error:", err.Error())
	}

	if _, ok := l.(nnet.Layer); !ok {
		t.Error("constructor returns not Layer type")
	}
	if _, ok := l.(*Activation); !ok {
		t.Error("constructor returns not Activation layer")
	}
}

func TestActivation_Complex(t *testing.T) {
	nnet.ActivationsRegistry["testActivationDoubleValue"] = &testActivationDoubleValue{}

	cfg := LayerConfigActivation(&testActivationDoubleValue{})

	l, err := LayerConstructorActivation(cfg)
	if err != nil {
		t.Errorf("create layer error:", err.Error())
	}

	iw, ih, id := 10, 1, 1
	ow, oh, od := l.InitDataSizes(iw, ih, id)

	if iw != ow || ih != oh || id != od {
		t.Error("output sizes not equal input sizes")
	}

	inputs := &nnet.Data{}
	inputs.InitVectorRandom(iw, -1, 1)

	output := l.Activate(inputs)

	if l.GetOutput() != output {
		t.Error("getOutput returns other result than activate")
	}

	if output.Dims[0] != ow || output.Dims[1] != oh || output.Dims[2] != od {
		t.Error("output dimentions mistmatch")
	}

	for i := 0; i < len(inputs.Data); i++ {
		if output.Data[i] != 2*inputs.Data[i] {
			t.Error("output value is invalid")
		}
	}

	deltas := &nnet.Data{}
	deltas.InitVector(iw)
	deltas.Fill(1)

	gradient := l.Backprop(deltas)

	for i := 0; i < iw; i++ {
		if gradient.Data[i] != inputs.Data[i] {
			t.Error("gradient value is invalid")
		}
	}

	config := l.Serialize()

	if config.Type != LAYER_ACTIVATION {
		t.Error("invalid layer type in serialized config")
	}

	actCode := config.Data.String(nnet.KEY_ACTIVATION)

	if actCode != "testActivationDoubleValue" {
		t.Error("missed ActCode in serialized config data")
	}
}