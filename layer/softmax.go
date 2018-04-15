package layer

import (
	"math"
	"fmt"
	"github.com/drdreyworld/nnet"
)

const LAYER_SOFTMAX = "softmax"

func init() {
	nnet.Layers[LAYER_SOFTMAX] = SoftmaxLayerConstructor
}

func SoftmaxLayerConstructor() nnet.Layer {
	return &Softmax{}
}

type Softmax struct {
	iWidth  int
	iHeight int
	iDepth  int

	oWidth  int
	oHeight int
	oDepth  int

	inputs *nnet.Mem
	output *nnet.Mem
}

func (l *Softmax) Init(config nnet.LayerConfig) (err error) {
	l.inputs = &nnet.Mem{}
	l.output = &nnet.Mem{}

	return
}

func (l *Softmax) InitDataSizes(w, h, d int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d
	l.oWidth, l.oHeight, l.oDepth = w, h, d

	l.output.InitTensor(w, h, d)

	fmt.Println("softmax output params:", w, h, d)

	return w, h, d
}

func (l *Softmax) Activate(inputs *nnet.Mem) *nnet.Mem {
	// inputs is readonly for layer
	l.inputs = inputs

	maxv := l.inputs.Data[0]
	summ := 0.0

	for i := 1; i < len(l.inputs.Data); i++ {
		if maxv < l.inputs.Data[i] {
			maxv = l.inputs.Data[i]
		}
	}

	for i := 0; i < len(l.inputs.Data); i++ {
		l.output.Data[i] = math.Exp(l.inputs.Data[i]-maxv)
		summ += l.output.Data[i]
	}

	for i := 0; i < len(l.inputs.Data); i++ {
		l.output.Data[i] = l.output.Data[i]/summ
	}

	// output is readonly for next layer
	return l.output
}

func (l *Softmax) GetOutput() *nnet.Mem {
	return l.output
}

func (l *Softmax) Backprop(deltas *nnet.Mem) *nnet.Mem {
	// deltas calculated in net.Backprop
	return deltas.Copy()
}

func (l *Softmax) Serialize() (res nnet.LayerConfig) {
	res.Type = LAYER_SOFTMAX
	return
}

func (l *Softmax) UnmarshalConfigDataFromJSON(b []byte) (interface{}, error) {
	return map[string]interface{}{}, nil
}