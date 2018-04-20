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

func testCreateActivationLayer(t *testing.T) *Activation {
	t.Helper()
	l := LayerConstructorActivation().(*Activation)
	l.ActFunc = "testActivationDoubleValue"

	return l
}

func TestLayerConstructorActivation(t *testing.T) {
	l := LayerConstructorActivation()
	if _, ok := l.(*Activation); !ok {
		t.Error("constructor returns not Activation layer")
	}
}

func TestActivation_InitDataSizes(t *testing.T) {
	l := testCreateActivationLayer(t)

	iw, ih, id := 10, 1, 1
	ow, oh, od := l.InitDataSizes(iw, ih, id)

	if iw != ow || ih != oh || id != od {
		t.Error("output sizes not equal input sizes")
	}
}

func TestActivation_Activate(t *testing.T) {
	l := testCreateActivationLayer(t)
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
	l := testCreateActivationLayer(t)
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