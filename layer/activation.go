package layer

import (
	"encoding/gob"
	"github.com/drdreyworld/nnet"
)

const LAYER_ACTIVATION = "activation"

func init() {
	nnet.LayersRegistry[LAYER_ACTIVATION] = LayerConstructorActivation
	gob.Register(Activation{})
}

func LayerConstructorActivation() nnet.Layer {
	return &Activation{}
}

type Activation struct {
	iWidth  int
	iHeight int
	iDepth  int

	inputs *nnet.Data
	output *nnet.Data

	ActFunc   string
	ActParams interface{}

	actFunc nnet.Activation
}

func (l *Activation) GetType() string {
	return LAYER_ACTIVATION
}

func (l *Activation) InitDataSizes(w, h, d int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d

	l.output = &nnet.Data{}
	l.output.InitCube(w, h, d)

	if f, ok := nnet.ActivationsRegistry[l.ActFunc]; ok {
		l.actFunc = f(l.ActParams)
	} else {
		panic("activation function is not registered:" + l.ActFunc)
	}

	return w, h, d
}

func (l *Activation) Activate(inputs *nnet.Data) *nnet.Data {
	l.inputs = inputs

	for i := 0; i < len(l.inputs.Data); i++ {
		l.output.Data[i] = l.actFunc.Forward(l.inputs.Data[i])
	}

	return l.output
}

func (l *Activation) Backprop(deltas *nnet.Data) (gradient *nnet.Data) {
	gradient = l.inputs.CopyZero()
	for i := 0; i < len(gradient.Data); i++ {
		gradient.Data[i] = deltas.Data[i] * l.actFunc.Backward(l.output.Data[i])
	}
	return
}

func (l *Activation) GetOutput() *nnet.Data {
	return l.output
}
