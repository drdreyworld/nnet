package layer

import (
	"testing"
	"github.com/drdreyworld/nnet"
)

type testActivationDoubleValue struct{}

func init() {
	nnet.ActivationsRegistry["testActivationDoubleValue"] = &testActivationDoubleValue{}
}

func (a *testActivationDoubleValue) Forward(v float64) float64 {
	return 2 * v
}

func (a *testActivationDoubleValue) Backward(v float64) float64 {
	return 0.5 * v
}

func (a *testActivationDoubleValue) Serialize() string {
	return "testActivationDoubleValue"
}

func createTestActivationLayerConfig(t *testing.T) nnet.LayerConfig {
	t.Helper()
	return LayerConfigActivation(&testActivationDoubleValue{})
}

func createTestActivationLayer(t *testing.T) nnet.Layer  {
	t.Helper()

	cfg := createTestActivationLayerConfig(t)

	l, err := LayerConstructorActivation(cfg)
	if err != nil {
		t.Error("create layer error:", err.Error())
	}

	return l
}

func TestLayerConfigActivation(t *testing.T) {
	cfg := createTestActivationLayerConfig(t)

	if err := cfg.CheckType(LAYER_ACTIVATION); err != nil {
		t.Error("config type invalid:", err.Error())
	}

	if cfg.Data == nil {
		t.Error("config data not initialized")
	}

	if _, ok := cfg.Data.GetActivation().(*testActivationDoubleValue); !ok {
		t.Error("invalid activation function in config")
	}
}

func TestLayerConstructorActivation(t *testing.T) {
	l := createTestActivationLayer(t)

	if _, ok := l.(nnet.Layer); !ok {
		t.Error("constructor returns not Layer type")
	}

	if _, ok := l.(*Activation); !ok {
		t.Error("constructor returns not Activation layer")
	}
}

func TestActivation_InitDataSizes(t *testing.T) {
	l := createTestActivationLayer(t)

	iw, ih, id := 10, 1, 1
	ow, oh, od := l.InitDataSizes(iw, ih, id)

	if iw != ow || ih != oh || id != od {
		t.Error("output sizes not equal input sizes")
	}
}

func TestActivation_Activate(t *testing.T) {
	l := createTestActivationLayer(t)
	a := testActivationDoubleValue{}

	iw, ih, id := 10, 1, 1
	ow, oh, od := l.InitDataSizes(iw, ih, id)

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
		if output.Data[i] != a.Forward(inputs.Data[i]) {
			t.Error("output value is invalid")
		}
	}
}

func TestActivation_Backprop(t *testing.T) {
	l := createTestActivationLayer(t)
	a := testActivationDoubleValue{}

	iw, ih, id := 10, 1, 1

	l.InitDataSizes(iw, ih, id)

	inputs := &nnet.Data{}
	inputs.InitVectorRandom(iw, -1, 1)

	output := l.Activate(inputs)

	deltas := &nnet.Data{}
	deltas.InitVector(iw)
	deltas.Fill(1)

	gradient := l.Backprop(deltas)

	for i := 0; i < iw; i++ {
		if gradient.Data[i] != deltas.Data[i] * a.Backward(output.Data[i]) {
			t.Error("gradient value is invalid")
		}
	}
}

func TestActivation_Serialize(t *testing.T) {
	l := createTestActivationLayer(t)
	c := l.Serialize()
	a := c.Data.GetActivation()

	if c.CheckType(LAYER_ACTIVATION) != nil {
		t.Error("invalid layer type in serialized config")
	}

	if a == nil {
		t.Error("missed activation in serialized config data")
	}

	if _, ok := a.(*testActivationDoubleValue); !ok {
		t.Error("invalid activation in serialized config data")
	}
}

func TestActivation_Unserialize(t *testing.T) {
	l := createTestActivationLayer(t)
	c := createTestActivationLayerConfig(t)

	if err := l.Unserialize(c); err != nil {
		t.Error("error on unserialization layer from config:", err.Error())
	}
}