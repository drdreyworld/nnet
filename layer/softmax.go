package layer

import (
	"github.com/drdreyworld/nnet"
	"math"
)

const LAYER_SOFTMAX = "softmax"

func init() {
	nnet.Layers[LAYER_SOFTMAX] = SoftmaxLayerConstructor
}

func SoftmaxLayerConstructor() nnet.Layer {
	return &Softmax{}
}

type Softmax struct {
	iWidth, iHeight, iDepth int
	oWidth, oHeight, oDepth int

	inputs *nnet.Data
	output *nnet.Data
}

func (l *Softmax) Init(config nnet.LayerConfig) (err error) {
	l.inputs = &nnet.Data{}
	l.output = &nnet.Data{}

	return
}

func (l *Softmax) InitDataSizes(w, h, d int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d
	l.oWidth, l.oHeight, l.oDepth = w, h, d

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

func (l *Softmax) Serialize() (res nnet.LayerConfig) {
	res.Type = LAYER_SOFTMAX
	return
}
