package layer

import (
	"encoding/gob"
	"github.com/drdreyworld/nnet"
	"log"
	"math"
	"math/rand"
	"time"
)

const LAYER_DENSE = "dense"

func init() {
	nnet.LayersRegistry[LAYER_DENSE] = LayerConstructorDense
	gob.Register(Dense{})
}

func LayerConstructorDense() nnet.Layer {
	return &Dense{}
}

type Dense struct {
	IWidth, IHeight, IDepth int
	OWidth, OHeight, ODepth int

	Weights *nnet.Data
	Biases  *nnet.Data

	inputs *nnet.Data
	output *nnet.Data

	gradWeights *nnet.Data
	gradBiases  *nnet.Data
	gradInputs  *nnet.Data
}

func (l *Dense) GetType() string {
	return LAYER_DENSE
}

func (l *Dense) InitDataSizes(w, h, d int) (oW, oH, oD int) {
	l.output = &nnet.Data{}
	l.output.InitCube(l.OWidth, l.OHeight, l.ODepth)

	l.IWidth, l.IHeight, l.IDepth = w, h, d

	if l.Weights == nil {
		l.Weights = &nnet.Data{}
		l.Biases = &nnet.Data{}
	}

	if len(l.Weights.Data) == 0 {
		rand.Seed(time.Now().UnixNano())
		maxWeight := math.Sqrt(1.0 / float64(l.IWidth*l.IHeight*l.IDepth))

		l.Biases.InitCube(l.OWidth, l.OHeight, l.ODepth)
		l.Weights.InitHiperCubeRandom(l.IWidth, l.IHeight, l.IDepth, l.OWidth*l.OHeight*l.ODepth, 0, maxWeight)
	}

	l.gradInputs = &nnet.Data{}
	l.gradInputs.InitCube(l.IWidth, l.IHeight, l.IDepth)

	l.gradBiases = &nnet.Data{}
	l.gradBiases.InitCube(l.OWidth, l.OHeight, l.ODepth)

	l.gradWeights = &nnet.Data{}
	l.gradWeights.InitHiperCube(l.IWidth, l.IHeight, l.IDepth, l.OWidth*l.OHeight*l.ODepth)

	log.Println("init layer: dense, input sizes:", w, h, d, "output sizes:", l.OWidth, l.OHeight, l.ODepth)

	return l.OWidth, l.OHeight, l.ODepth
}

func (l *Dense) Activate(inputs *nnet.Data) *nnet.Data {
	l.inputs = inputs
	iVolume := l.IWidth * l.IHeight * l.IDepth

	for i := 0; i < len(l.output.Data); i++ {
		k := i * iVolume
		o := l.Biases.Data[i]

		for j := 0; j < len(l.inputs.Data); j++ {
			o += l.Weights.Data[k+j] * l.inputs.Data[j]
		}

		l.output.Data[i] = o
	}
	return l.output
}

func (l *Dense) Backprop(deltas *nnet.Data) *nnet.Data {
	iVolume := l.IWidth * l.IHeight * l.IDepth

	for i := 0; i < len(l.output.Data); i++ {
		k := i * iVolume

		l.gradBiases.Data[i] += deltas.Data[i]

		for j := 0; j < len(l.inputs.Data); j++ {
			l.gradInputs.Data[j] += deltas.Data[i] * l.Weights.Data[k+j]
			l.gradWeights.Data[k+j] += l.inputs.Data[j] * deltas.Data[i]
		}
	}

	return l.gradInputs
}

func (l *Dense) GetOutput() *nnet.Data {
	return l.output
}

func (l *Dense) ResetGradients() {
	l.gradInputs.Fill(0)
	l.gradBiases.Fill(0)
	l.gradWeights.Fill(0)
}

func (l *Dense) GetWeightsWithGradient() (*nnet.Data, *nnet.Data) {
	return l.Weights, l.gradWeights
}

func (l *Dense) GetBiasesWithGradient() (*nnet.Data, *nnet.Data) {
	return l.Biases, l.gradBiases
}
