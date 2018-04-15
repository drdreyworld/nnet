package layer

import (
	"fmt"
	"github.com/drdreyworld/nnet"
)

const LAYER_RELU = "relu"

func init() {
	nnet.Layers[LAYER_RELU] = ReluLayerConstructor
}

func ReluLayerConstructor() nnet.Layer {
	return &Relu{}
}

type Relu struct {
	iWidth  int
	iHeight int
	iDepth  int

	inputs *nnet.Mem
	output *nnet.Mem
}

func (l *Relu) Init(config nnet.LayerConfig) (err error) {
	l.output = &nnet.Mem{}
	return
}

func (l *Relu) InitDataSizes(w, h, d int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d
	l.output.InitTensor(w, h, d)

	fmt.Println("relu output params:", w, h, d)

	return w, h, d
}

func (l *Relu) Activate(inputs *nnet.Mem) *nnet.Mem {
	// inputs is readonly for layer
	l.inputs = inputs
	for i := 0; i < len(l.inputs.Data); i++ {
		if l.inputs.Data[i] > 0 {
			l.output.Data[i] = l.inputs.Data[i]
		} else {
			l.output.Data[i] = 0
		}
	}
	// output is readonly for next layer
	return l.output
}

func (l *Relu) Backprop(deltas *nnet.Mem) (gradient *nnet.Mem) {
	gradient = deltas.Copy()
	for i := 0; i < len(gradient.Data); i++ {
		if l.output.Data[i] <= 0 {
			gradient.Data[i] = 0
		}
	}
	return
}

func (l *Relu) Serialize() (res nnet.LayerConfig) {
	res.Type = LAYER_RELU
	return
}

func (l *Relu) UnmarshalConfigDataFromJSON(b []byte) (interface{}, error) {
	return nil, nil
}

func (l *Relu) GetOutput() *nnet.Mem {
	return l.output
}
