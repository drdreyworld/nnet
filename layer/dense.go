package layer

import (
	"github.com/drdreyworld/nnet"
	"math"
	"math/rand"
	"time"
)

const LAYER_DENSE = "dense"

func init() {
	nnet.Layers[LAYER_DENSE] = DenseLayerConstructor
}

func DenseLayerConstructor() nnet.Layer {
	return &Dense{}
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

func (l *Dense) Init(config nnet.LayerConfig) (err error) {
	l.output = &nnet.Data{}
	l.gradWeights = &nnet.Data{}
	l.gradBiases = &nnet.Data{}
	l.gradInputs = &nnet.Data{}

	l.oWidth = config.Data.Int("OWidth")
	l.oHeight = config.Data.Int("OHeight")
	l.oDepth = config.Data.Int("ODepth")

	l.output.InitCube(l.oWidth, l.oHeight, l.oDepth)

	if w, ok := config.Data["Weights"].(nnet.Data); ok {
		l.weights = &w
	} else {
		l.weights = &nnet.Data{}
	}

	if w, ok := config.Data["Biases"].(nnet.Data); ok {
		l.biases = &w
	} else {
		l.biases = &nnet.Data{}
	}

	return
}

func (l *Dense) InitDataSizes(w, h, d int) (oW, oH, oD int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d

	if len(l.weights.Data) == 0 {
		rand.Seed(time.Now().UnixNano())
		maxWeight := math.Sqrt(1.0 / float64(l.iWidth*l.iHeight*l.iDepth))

		l.biases.InitCube(l.oWidth, l.oHeight, l.oDepth)
		l.weights.InitHiperCubeRandom(l.iWidth, l.iHeight, l.iDepth, l.oWidth*l.oHeight*l.oDepth, 0, maxWeight)
	}

	l.gradInputs.InitCube(l.iWidth, l.iHeight, l.iDepth)
	l.gradBiases.InitCube(l.oWidth, l.oHeight, l.oDepth)
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

func (l *Dense) Serialize() (res nnet.LayerConfig) {
	res.Type = LAYER_DENSE
	res.Data = nnet.LayerConfigData{
		"OWidth":  l.oWidth,
		"OHeight": l.oHeight,
		"ODepth":  l.oDepth,
	}

	if l.weights != nil {
		res.Data["Weights"] = *l.weights
		res.Data["Biases"] = *l.biases
	}

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
