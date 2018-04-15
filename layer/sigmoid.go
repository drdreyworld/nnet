package layer

import (
	"fmt"
	"github.com/drdreyworld/nnet"
	"github.com/drdreyworld/nnet/activation"
)

const LAYER_SIGMOID = "sigmoid"

func init() {
	nnet.Layers[LAYER_SIGMOID] = SigmoidLayerConstructor
}

func SigmoidLayerConstructor() nnet.Layer {
	return &Sigmoid{}
}

type Sigmoid struct {
	iWidth  int
	iHeight int
	iDepth  int

	inputs *nnet.Mem
	output *nnet.Mem
	actfun nnet.Activation
}

func (l *Sigmoid) Init(config nnet.LayerConfig) (err error) {
	l.actfun = nnet.Activations[nn.ACTIVATION_SIGMOID]
	l.output = &nnet.Mem{}
	return
}

func (l *Sigmoid) InitDataSizes(w, h, d int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d
	l.output.InitTensor(w, h, d)

	fmt.Println("sigmoid output params:", w, h, d)

	return w, h, d
}

func (l *Sigmoid) Activate(inputs *nnet.Mem) *nnet.Mem {
	// inputs is readonly for layer
	l.inputs = inputs
	for i := 0; i < len(l.inputs.Data); i++ {
		l.output.Data[i] = l.actfun.Forward(l.inputs.Data[i])
	}
	// output is readonly for next layer
	return l.output
}

func (l *Sigmoid) Backprop(deltas *nnet.Mem) (gradient *nnet.Mem) {
	gradient = l.inputs.CopyZero()
	for i := 0; i < len(gradient.Data); i++ {
		gradient.Data[i] = deltas.Data[i] * l.actfun.Backward(l.output.Data[i])
	}
	return
}

func (l *Sigmoid) Serialize() (res nnet.LayerConfig) {
	res.Type = LAYER_SIGMOID
	return
}

func (l *Sigmoid) UnmarshalConfigDataFromJSON(b []byte) (interface{}, error) {
	return nil, nil
}

func (l *Sigmoid) GetOutput() *nnet.Mem {
	return l.output
}
