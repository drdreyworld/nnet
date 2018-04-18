package layer

import (
	"github.com/drdreyworld/nnet"
	"math"
	"math/rand"
	"time"
)

const LAYER_DENSE = "dense"

func init() {
	nnet.LayersRegistry[LAYER_DENSE] = LayerConstructorDense
}

func LayerConfigDense(OWidth, OHeight, ODepth int) (res nnet.LayerConfig) {
	res.Type = LAYER_DENSE
	res.Data = nnet.LayerConfigData{
		"OWidth":  OWidth,
		"OHeight": OHeight,
		"ODepth":  ODepth,
	}
	return
}

func LayerConstructorDense(cfg nnet.LayerConfig) (res nnet.Layer, err error) {
	res = &Dense{}
	err = res.Unserialize(cfg)
	return
}

type Dense struct {
	iWidth, iHeight, iDepth int
	oWidth, oHeight, oDepth int

	weights *nnet.Data

	inputs *nnet.Data
	output *nnet.Data
	biases *nnet.Data

	gradWeights *nnet.Data
	gradBiases  *nnet.Data
	gradInputs  *nnet.Data
}

func (l *Dense) InitDataSizes(w, h, d int) (oW, oH, oD int) {
	l.output = &nnet.Data{}
	l.output.InitCube(l.oWidth, l.oHeight, l.oDepth)

	l.iWidth, l.iHeight, l.iDepth = w, h, d

	if len(l.weights.Data) == 0 {
		rand.Seed(time.Now().UnixNano())
		maxWeight := math.Sqrt(1.0 / float64(l.iWidth*l.iHeight*l.iDepth))

		l.biases.InitCube(l.oWidth, l.oHeight, l.oDepth)
		l.weights.InitHiperCubeRandom(l.iWidth, l.iHeight, l.iDepth, l.oWidth*l.oHeight*l.oDepth, 0, maxWeight)
	}

	l.gradInputs = &nnet.Data{}
	l.gradInputs.InitCube(l.iWidth, l.iHeight, l.iDepth)

	l.gradBiases = &nnet.Data{}
	l.gradBiases.InitCube(l.oWidth, l.oHeight, l.oDepth)

	l.gradWeights = &nnet.Data{}
	l.gradWeights.InitHiperCube(l.iWidth, l.iHeight, l.iDepth, l.oWidth*l.oHeight*l.oDepth)

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *Dense) Activate(inputs *nnet.Data) *nnet.Data {
	l.inputs = inputs
	iVolume := l.iWidth * l.iHeight * l.iDepth

	for i := 0; i < len(l.output.Data); i++ {
		k := i * iVolume
		o := l.biases.Data[i]

		for j := 0; j < len(l.inputs.Data); j++ {
			o += l.weights.Data[k+j] * l.inputs.Data[j]
		}

		l.output.Data[i] = o
	}
	return l.output
}

func (l *Dense) Backprop(deltas *nnet.Data) *nnet.Data {
	iVolume := l.iWidth * l.iHeight * l.iDepth

	for i := 0; i < len(l.output.Data); i++ {
		k := i * iVolume

		l.gradBiases.Data[i] += deltas.Data[i]

		for j := 0; j < len(l.inputs.Data); j++ {
			l.gradInputs.Data[j] += deltas.Data[i] * l.weights.Data[k+j]
			l.gradWeights.Data[k+j] += l.inputs.Data[j] * deltas.Data[i]
		}
	}

	return l.gradInputs
}

func (l *Dense) Unserialize(cfg nnet.LayerConfig) (err error) {
	if err = cfg.CheckType(LAYER_DENSE); err == nil {
		l.oWidth = cfg.Data.Int("OWidth")
		l.oHeight = cfg.Data.Int("OHeight")
		l.oDepth = cfg.Data.Int("ODepth")
		l.weights = cfg.Data.GetWeights()
		l.biases = cfg.Data.GetBiases()
	}
	return
}

func (l *Dense) Serialize() (res nnet.LayerConfig) {
	res = LayerConfigDense(l.oWidth, l.oHeight, l.oDepth)
	res.Data.SetWeights(l.weights)
	res.Data.SetBiases(l.biases)

	return
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
	return l.weights, l.gradWeights
}

func (l *Dense) GetBiasesWithGradient() (*nnet.Data, *nnet.Data) {
	return l.biases, l.gradBiases
}
