package layer

import (
	"github.com/drdreyworld/nnet"
	"math"
	"math/rand"
	"time"
)

const LAYER_CONV = "conv"

func init() {
	nnet.LayersRegistry[LAYER_CONV] = LayerConstructorConv
}

func LayerConfigConv(FWidth, FHeight, FDepth, FPadding, FStride int) (res nnet.LayerConfig) {
	res.Type = LAYER_CONV
	res.Data = nnet.LayerConfigData{
		"FWidth":  FWidth,
		"FHeight": FHeight,
		"FDepth":  FDepth,
		"Padding": FPadding,
		"Stride":  FStride,
	}
	return
}

func LayerConstructorConv(cfg nnet.LayerConfig) (res nnet.Layer, err error) {
	res = &Conv{}
	err = res.Unserialize(cfg)
	return
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

func (l *Conv) InitDataSizes(w, h, d int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d

	l.oWidth = (l.iWidth-l.fwidth+2*l.fpadding)/l.fstride + 1
	l.oHeight = (l.iHeight-l.fheight+2*l.fpadding)/l.fstride + 1

	l.output = &nnet.Data{}
	l.output.InitCube(l.oWidth, l.oHeight, l.oDepth)

	if len(l.weights.Data) == 0 {
		rand.Seed(time.Now().UnixNano())
		maxWeight := math.Sqrt(1.0 / float64(l.fwidth*l.fheight*l.oDepth*l.iDepth))

		l.weights.InitCubeRandom(l.fwidth, l.fheight, l.oDepth*l.iDepth, -maxWeight, maxWeight)
		l.biases.InitVector(l.oDepth)
	}

	l.gradBiases = &nnet.Data{}
	l.gradBiases.InitVector(l.oDepth)

	l.gradWeights = &nnet.Data{}
	l.gradWeights.InitCube(l.fwidth, l.fheight, l.iDepth*l.oDepth)

	l.gradInputs = &nnet.Data{}
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


func (l *Conv) Unserialize(cfg nnet.LayerConfig) (err error) {
	if err = cfg.CheckType(LAYER_CONV); err == nil {
		l.fwidth = cfg.Data.Int("FWidth")
		l.fheight = cfg.Data.Int("FHeight")
		l.oDepth = cfg.Data.Int("FDepth")

		l.fpadding = cfg.Data.Int("Padding")
		l.fstride = cfg.Data.Int("Stride")

		if l.fstride < 1 {
			l.fstride = 1
		}

		l.weights = cfg.Data.GetWeights()
		l.biases = cfg.Data.GetBiases()
	}
	return
}


func (l *Conv) Serialize() (res nnet.LayerConfig) {
	res = LayerConfigConv(l.fwidth, l.fheight, l.oDepth, l.fpadding, l.fstride)
	res.Data.SetWeights(l.weights)
	res.Data.SetBiases(l.biases)

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
