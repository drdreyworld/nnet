package layer

import (
	"github.com/drdreyworld/nnet"
	"math"
)

const LAYER_SOFTMAX = "softmax"

func init() {
	nnet.LayersRegistry[LAYER_SOFTMAX] = LayerConstructorSoftmax
}

func LayerConfigSoftmax() (res nnet.LayerConfig) {
	res.Type = LAYER_SOFTMAX
	return
}

func LayerConstructorSoftmax(cfg nnet.LayerConfig) (res nnet.Layer, err error) {
	res = &Softmax{}
	err = res.Unserialize(cfg)
	return
}

type Softmax struct {
	iWidth, iHeight, iDepth int
	oWidth, oHeight, oDepth int

	inputs *nnet.Data
	output *nnet.Data
}

func (l *Softmax) InitDataSizes(w, h, d int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d
	l.oWidth, l.oHeight, l.oDepth = w, h, d

	l.output = &nnet.Data{}
	l.output.InitCube(w, h, d)

	return w, h, d
}

func (l *Softmax) Activate(inputs *nnet.Data) *nnet.Data {
	l.inputs = inputs

	maxv := l.inputs.Data[0]
	summ := 0.0

	for i := 1; i < len(l.inputs.Data); i++ {
		if maxv < l.inputs.Data[i] {
			maxv = l.inputs.Data[i]
		}
	}

	for i := 0; i < len(l.inputs.Data); i++ {
		l.output.Data[i] = math.Exp(l.inputs.Data[i] - maxv)
		summ += l.output.Data[i]
	}

	for i := 0; i < len(l.inputs.Data); i++ {
		l.output.Data[i] = l.output.Data[i] / summ
	}
	return l.output
}

func (l *Softmax) GetOutput() *nnet.Data {
	return l.output
}

func (l *Softmax) Backprop(deltas *nnet.Data) *nnet.Data {
	return deltas.Copy()
}

func (l *Softmax) Unserialize(cfg nnet.LayerConfig) error {
	return cfg.CheckType(LAYER_SOFTMAX)
}

func (l *Softmax) Serialize() (res nnet.LayerConfig) {
	return LayerConfigSoftmax()
}
