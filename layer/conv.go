package layer

import (
	"github.com/drdreyworld/nnet"
	"math"
	"math/rand"
	"time"
)

const LAYER_CONV = "conv"

func init() {
	nnet.Layers[LAYER_CONV] = ConvLayerConstructor
}

func ConvLayerConstructor() nnet.Layer {
	return &Conv{}
}

type Conv struct {
	iWidth, iHeight, iDepth int
	oWidth, oHeight, oDepth int

	fwidth  int
	fheight int

	fpadding int
	fstride  int

	weights *nnet.Data
	inputs  *nnet.Data
	output  *nnet.Data
	biases  *nnet.Data

	gradWeights *nnet.Data
	gradBiases  *nnet.Data
	gradInputs  *nnet.Data
}

func (l *Conv) Init(config nnet.LayerConfig) (err error) {
	l.output = &nnet.Data{}
	l.gradWeights = &nnet.Data{}
	l.gradBiases = &nnet.Data{}
	l.gradInputs = &nnet.Data{}

	l.fwidth = config.Data.Int("FWidth")
	l.fheight = config.Data.Int("FHeight")
	l.oDepth = config.Data.Int("FDepth")

	l.fpadding = config.Data.Int("Padding")
	l.fstride = config.Data.Int("Stride")

	if l.fstride < 1 {
		l.fstride = 1
	}

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

func (l *Conv) InitDataSizes(w, h, d int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d

	l.oWidth = (l.iWidth-l.fwidth+2*l.fpadding)/l.fstride + 1
	l.oHeight = (l.iHeight-l.fheight+2*l.fpadding)/l.fstride + 1

	l.output.InitCube(l.oWidth, l.oHeight, l.oDepth)

	if len(l.weights.Data) == 0 {
		rand.Seed(time.Now().UnixNano())
		maxWeight := math.Sqrt(1.0 / float64(l.fwidth*l.fheight*l.oDepth*l.iDepth))

		l.weights.InitCubeRandom(l.fwidth, l.fheight, l.oDepth*l.iDepth, -maxWeight, maxWeight)
		l.biases.InitVector(l.oDepth)
	}

	l.gradBiases.InitVector(l.oDepth)
	l.gradWeights.InitCube(l.fwidth, l.fheight, l.iDepth*l.oDepth)
	l.gradInputs.InitCube(l.iWidth, l.iHeight, l.iDepth)

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *Conv) Activate(inputs *nnet.Data) *nnet.Data {
	l.inputs = inputs

	outXYZ := 0
	iSquare := l.iWidth * l.iHeight
	wSquare := l.fwidth * l.fheight
	wQube := l.iDepth * wSquare

	for fi := 0; fi < l.oDepth; fi++ {
		wi := fi * wQube
		for oy, initInputY := 0, -l.fpadding; oy < l.oHeight; oy, initInputY = oy+1, initInputY+l.fstride {
			for ox, initInputX := 0, -l.fpadding; ox < l.oWidth; ox, initInputX = ox+1, initInputX+l.fstride {

				l.output.Data[outXYZ] = l.biases.Data[fi]

				for fy, iy := 0, initInputY; fy < l.fheight; fy, iy = fy+1, iy+1 {
					for fx, ix := 0, initInputX; iy > -1 && iy < l.iHeight && fx < l.fwidth; fx, ix = fx+1, ix+1 {
						for iz := 0; ix > -1 && ix < l.iWidth && iz < l.iDepth; iz++ {
							inXYZ := iz*iSquare + iy*l.iWidth + ix
							wtXYZ := wi + iz*wSquare + fy*l.fwidth + fx

							l.output.Data[outXYZ] += l.weights.Data[wtXYZ] * l.inputs.Data[inXYZ]
						}
					}
				}

				outXYZ++
			}
		}
	}

	return l.output
}

func (l *Conv) Backprop(deltas *nnet.Data) *nnet.Data {
	weights := l.weights.Copy()
	weights.RotateMatrixesInCube()

	outXYZ := 0
	iSquare := l.iWidth * l.iHeight
	wSquare := l.fwidth * l.fheight
	wQube := l.iDepth * wSquare

	for fi := 0; fi < l.oDepth; fi++ {
		wi := fi * wQube
		for oy, initInputY := 0, -l.fpadding; oy < l.oHeight; oy, initInputY = oy+1, initInputY+l.fstride {
			for ox, initInputX := 0, -l.fpadding; ox < l.oWidth; ox, initInputX = ox+1, initInputX+l.fstride {

				l.gradBiases.Data[fi] += deltas.Data[outXYZ]

				for fy, iy := 0, initInputY; fy < l.fheight; fy, iy = fy+1, iy+1 {
					for fx, ix := 0, initInputX; iy > -1 && iy < l.iHeight && fx < l.fwidth; fx, ix = fx+1, ix+1 {
						for iz := 0; ix > -1 && ix < l.iWidth && iz < l.iDepth; iz++ {
							inXYZ := iz*iSquare + iy*l.iWidth + ix
							wtXYZ := wi + iz*wSquare + fy*l.fwidth + fx
							l.gradInputs.Data[inXYZ] += weights.Data[wtXYZ] * deltas.Data[outXYZ]
							l.gradWeights.Data[wtXYZ] += l.inputs.Data[inXYZ] * deltas.Data[outXYZ]
						}
					}
				}

				outXYZ++
			}
		}
	}

	return l.gradInputs
}

func (l *Conv) Serialize() (res nnet.LayerConfig) {
	res.Type = LAYER_CONV
	res.Data = nnet.LayerConfigData{
		"FWidth":  l.fwidth,
		"FHeight": l.fheight,
		"FDepth":  l.oDepth,
		"Padding": l.fpadding,
		"Stride":  l.fstride,
	}

	if l.weights != nil {
		res.Data["Weights"] = *l.weights
		res.Data["Biases"] = *l.biases
	}
	return
}

func (l *Conv) GetWeights() *nnet.Data {
	return l.weights
}

func (l *Conv) GetOutput() *nnet.Data {
	return l.output
}

func (l *Conv) ResetGradients() {
	l.gradBiases.Fill(0)
	l.gradWeights.Fill(0)
	l.gradInputs.Fill(0)
}

func (l *Conv) GetWeightsWithGradient() (*nnet.Data, *nnet.Data) {
	return l.weights, l.gradWeights
}

func (l *Conv) GetBiasesWithGradient() (*nnet.Data, *nnet.Data) {
	return l.biases, l.gradBiases
}
