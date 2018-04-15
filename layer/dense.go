package layer

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand"
	"time"
	"fmt"
	"github.com/drdreyworld/nnet"
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

	weights *nnet.Mem

	inputs *nnet.Mem
	output *nnet.Mem
	biases *nnet.Mem

	gradWeights *nnet.Mem
	gradBiases  *nnet.Mem
	gradInputs  *nnet.Mem
}

func (l *Dense) Init(config nnet.LayerConfig) (err error) {
	l.output = &nnet.Mem{}
	l.biases = &nnet.Mem{}
	l.weights = &nnet.Mem{}
	l.gradWeights = &nnet.Mem{}
	l.gradBiases = &nnet.Mem{}
	l.gradInputs = &nnet.Mem{}

	if config.Data == nil {
		return errors.New("Config data is missed")
	}

	c, ok := config.Data.(DenseConfig)
	if !ok {
		panic("Invalid config for dense layer")
		return errors.New("Invalid config for dense layer")
	}

	if err = c.Check(); err != nil {
		panic(err)
		return
	}

	l.oWidth, l.oHeight, l.oDepth = c.OWidth, c.OHeight, c.ODepth
	l.output.InitTensor(l.oWidth, l.oHeight, l.oDepth)

	if len(c.Weights.Data) > 0 {
		l.weights = &c.Weights
		l.biases = &c.Biases
	}

	return
}

func (l *Dense) InitDataSizes(w, h, d int) (oW, oH, oD int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d

	if len(l.weights.Data) == 0 {
		rand.Seed(time.Now().UnixNano())
		maxWeight := math.Sqrt(1.0 / float64(l.iWidth*l.iHeight*l.iDepth))

		l.biases.InitTensor(l.oWidth, l.oHeight, l.oDepth)
		l.weights.InitHiperCubeRandom(l.iWidth, l.iHeight, l.iDepth, l.oWidth*l.oHeight*l.oDepth, 0, maxWeight)
	}

	l.gradInputs.InitTensor(l.iWidth, l.iHeight, l.iDepth)
	l.gradBiases.InitTensor(l.oWidth, l.oHeight, l.oDepth)
	l.gradWeights.InitHiperCube(l.iWidth, l.iHeight, l.iDepth, l.oWidth*l.oHeight*l.oDepth)

	fmt.Println("dense output params:", l.oWidth, l.oHeight, l.oDepth)

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *Dense) Activate(inputs *nnet.Mem) *nnet.Mem {
	// inputs is readonly for layer
	l.inputs = inputs
	ivolume := l.iWidth * l.iHeight * l.iDepth

	for i := 0; i < len(l.output.Data); i++ {
		k := i * ivolume
		o := l.biases.Data[i]

		for j := 0; j < len(l.inputs.Data); j++ {
			o += l.weights.Data[k + j] * l.inputs.Data[j]
		}

		l.output.Data[i] = o
	}

	// output is readonly for next layer
	return l.output
}

func (l *Dense) Backprop(deltas *nnet.Mem) *nnet.Mem {
	ivolume := l.iWidth * l.iHeight * l.iDepth

	for i := 0; i < len(l.output.Data); i++ {
		k := i * ivolume

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
	res.Data = DenseConfig{
		OWidth:  l.oWidth,
		OHeight: l.oHeight,
		ODepth:  l.oDepth,

		Weights: *l.weights,
		Biases:  *l.biases,
	}
	return
}

func (l *Dense) UnmarshalConfigDataFromJSON(b []byte) (interface{}, error) {
	cfg := DenseConfig{}
	err := json.Unmarshal(b, &cfg)

	return cfg, err
}

func (l *Dense) GetOutput() *nnet.Mem {
	return l.output
}

func (l *Dense) ResetGradients() {
	l.gradInputs.Fill(0)
	l.gradBiases.Fill(0)
	l.gradWeights.Fill(0)
}

func (l *Dense) GetWeightsWithGradient() (*nnet.Mem, *nnet.Mem) {
	return l.weights, l.gradWeights
}

func (l *Dense) GetBiasesWithGradient() (*nnet.Mem, *nnet.Mem) {
	return l.biases, l.gradBiases
}
