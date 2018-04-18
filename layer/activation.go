package layer

import (
	"github.com/drdreyworld/nnet"
)

const LAYER_ACTIVATION = "activation"

func init() {
	nnet.LayersRegistry[LAYER_ACTIVATION] = LayerConstructorActivation
}

func LayerConfigActivation(activation nnet.Activation) (res nnet.LayerConfig) {
	res.Type = LAYER_ACTIVATION
	res.Data = nnet.LayerConfigData{}
	res.Data.SetActivation(activation)
	return
}

func LayerConstructorActivation(cfg nnet.LayerConfig) (res nnet.Layer, err error) {
	res = &Activation{}
	err = res.Unserialize(cfg)
	return
}

type Activation struct {
	iWidth  int
	iHeight int
	iDepth  int

	inputs *nnet.Data
	output *nnet.Data

	actFunc nnet.Activation
	actCode string
}

func (l *Activation) InitDataSizes(w, h, d int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d

	l.output = &nnet.Data{}
	l.output.InitCube(w, h, d)

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

func (l *Activation) Unserialize(cfg nnet.LayerConfig) (err error) {
	if err = cfg.CheckType(LAYER_ACTIVATION); err == nil {
		l.actFunc = cfg.Data.GetActivation()
	}
	return
}

func (l *Activation) Serialize() (res nnet.LayerConfig) {
	return LayerConfigActivation(l.actFunc)
}

func (l *Activation) GetOutput() *nnet.Data {
	return l.output
}
