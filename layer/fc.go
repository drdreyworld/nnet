package layer

import (
	"encoding/gob"
	"github.com/drdreyworld/nnet"
	"log"
	"math"
	"sync"
)

const LAYER_FC = "fc"

func init() {
	nnet.LayersRegistry[LAYER_FC] = LayerConstructorFC
	gob.Register(FC{})
}

func LayerConstructorFC() nnet.Layer {
	return &FC{}
}

type FC struct {
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

func (l *FC) GetType() string {
	return LAYER_FC
}

func (l *FC) InitDataSizes(w, h, d int) (oW, oH, oD int) {
	l.output = &nnet.Data{}
	l.output.InitCube(l.OWidth, l.OHeight, l.ODepth)

	l.IWidth, l.IHeight, l.IDepth = w, h, d

	if l.Weights == nil {
		l.Weights = &nnet.Data{}
		l.Biases = &nnet.Data{}
	}

	if len(l.Weights.Data) == 0 {
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

func (l *FC) Activate(inputs *nnet.Data) *nnet.Data {
	l.inputs = inputs
	iVolume := l.IWidth * l.IHeight * l.IDepth

	wg := new(sync.WaitGroup)
	wg.Add(len(l.output.Data))

	for i := 0; i < len(l.output.Data); i++ {
		go l.activateNeuron(wg, i, iVolume)
	}

	wg.Wait()
	return l.output
}

func (l *FC) activateNeuron(wg *sync.WaitGroup, i, iVolume int) {
	defer wg.Done()

	k := i * iVolume
	o := l.Biases.Data[i]

	for j := 0; j < len(l.inputs.Data); j++ {
		o += l.Weights.Data[k+j] * l.inputs.Data[j]
	}

	l.output.Data[i] = o
}

func (l *FC) calcNeuronDelta(wg *sync.WaitGroup, i, iVolume int, deltas *nnet.Data) {
	defer wg.Done()

	k := i * iVolume

	l.gradBiases.Data[i] += deltas.Data[i]

	for j := 0; j < len(l.inputs.Data); j++ {
		l.gradInputs.Data[j] += deltas.Data[i] * l.Weights.Data[k+j]
		l.gradWeights.Data[k+j] += l.inputs.Data[j] * deltas.Data[i]
	}
}

func (l *FC) Backprop(deltas *nnet.Data) *nnet.Data {
	iVolume := l.IWidth * l.IHeight * l.IDepth

	wg := new(sync.WaitGroup)
	wg.Add(len(l.output.Data))

	for i := 0; i < len(l.output.Data); i++ {
		go l.calcNeuronDelta(wg, i, iVolume, deltas)
	}

	wg.Wait()
	return l.gradInputs
}

func (l *FC) GetOutput() *nnet.Data {
	return l.output
}

func (l *FC) ResetGradients() {
	l.gradInputs.Reset()
	l.gradBiases.Reset()
	l.gradWeights.Reset()
}

func (l *FC) GetWeightsWithGradient() (*nnet.Data, *nnet.Data) {
	return l.Weights, l.gradWeights
}

func (l *FC) GetBiasesWithGradient() (*nnet.Data, *nnet.Data) {
	return l.Biases, l.gradBiases
}

func (l *FC) GetInputGradients() (g *nnet.Data) {
	return l.gradInputs
}

func (l *FC) Mutable() bool {
	return true
}
