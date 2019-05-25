package layer

import (
	"encoding/gob"
	"github.com/drdreyworld/nnet"
	"log"
	"math"
)

const LAYER_SOFTMAX = "softmax"

func init() {
	nnet.LayersRegistry[LAYER_SOFTMAX] = LayerConstructorSoftmax
	gob.Register(Softmax{})
}

func LayerConstructorSoftmax() nnet.Layer {
	return &Softmax{}
}

type Softmax struct {
	IWidth, IHeight, IDepth int
	OWidth, OHeight, ODepth int

	inputs *nnet.Data
	output *nnet.Data
}

func (l *Softmax) GetType() string {
	return LAYER_SOFTMAX
}

func (l *Softmax) InitDataSizes(w, h, d int) (int, int, int) {
	l.IWidth, l.IHeight, l.IDepth = w, h, d
	l.OWidth, l.OHeight, l.ODepth = w, h, d

	l.output = &nnet.Data{}
	l.output.InitCube(w, h, d)

	log.Println("init layer: softmax, input sizes:", w, h, d, "output sizes:", w, h, d)

	return l.OWidth, l.OHeight, l.ODepth
}

func (l *Softmax) Activate(inputs *nnet.Data) *nnet.Data {
	l.inputs = inputs

	summ := 0.0
	maxv := 0.0

	cnt := len(l.inputs.Data)

	for i := 0; i < cnt; i++ {
		if i == 0 || maxv < l.inputs.Data[i] {
			maxv = l.inputs.Data[i]
		}
	}

	for i := 0; i < cnt; i++ {
		l.output.Data[i] = math.Exp(l.inputs.Data[i] - maxv)
		summ += l.output.Data[i]
	}

	for i := 0; i < cnt; i++ {
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
