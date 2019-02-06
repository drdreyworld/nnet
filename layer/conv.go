package layer

import (
	"encoding/gob"
	"github.com/drdreyworld/nnet"
	"log"
	"math"
	"math/rand"
	"time"
)

const LAYER_CONV = "conv"

func init() {
	nnet.LayersRegistry[LAYER_CONV] = LayerConstructorConv
	gob.Register(Conv{})
}

func LayerConstructorConv() nnet.Layer {
	return &Conv{}
}

type Conv struct {
	iWidth, iHeight, iDepth int
	oWidth, oHeight, oDepth int

	FWidth  int
	FHeight int
	FCount  int

	FPadding int
	FStride  int

	Weights *nnet.Data
	Biases  *nnet.Data

	inputs *nnet.Data
	output *nnet.Data

	gradWeights *nnet.Data
	gradBiases  *nnet.Data
	gradInputs  *nnet.Data
}

func (l *Conv) GetType() string {
	return LAYER_CONV
}

func (l *Conv) InitDataSizes(w, h, d int) (int, int, int) {
	if l.FStride == 0 {
		l.FStride = 1
	}

	l.iWidth, l.iHeight, l.iDepth = w, h, d

	l.oWidth = (l.iWidth-l.FWidth+2*l.FPadding)/l.FStride + 1
	l.oHeight = (l.iHeight-l.FHeight+2*l.FPadding)/l.FStride + 1
	l.oDepth = l.FCount

	l.output = &nnet.Data{}
	l.output.InitCube(l.oWidth, l.oHeight, l.oDepth)

	if l.Weights == nil {
		l.Weights = &nnet.Data{}
		l.Biases = &nnet.Data{}
	}

	if len(l.Weights.Data) == 0 {
		rand.Seed(time.Now().UnixNano())
		maxWeight := math.Sqrt(1.0 / float64(l.FWidth*l.FHeight*l.oDepth*l.iDepth))

		l.Weights.InitCubeRandom(l.FWidth, l.FHeight, l.oDepth*l.iDepth, -maxWeight, maxWeight)
		l.Biases.InitVector(l.oDepth)
	}

	l.gradBiases = &nnet.Data{}
	l.gradBiases.InitVector(l.oDepth)

	l.gradWeights = &nnet.Data{}
	l.gradWeights.InitCube(l.FWidth, l.FHeight, l.iDepth*l.oDepth)

	l.gradInputs = &nnet.Data{}
	l.gradInputs.InitCube(l.iWidth, l.iHeight, l.iDepth)

	log.Println("init layer: conv,",
		"input sizes:", w, h, d,
		"output sizes:", l.oWidth, l.oHeight, l.oDepth,
		"matrix sizes:", l.FWidth, "x", l.FHeight,
		"count:", l.FCount,
	)

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *Conv) Activate(inputs *nnet.Data) *nnet.Data {
	l.inputs = inputs

	outXYZ := 0
	iSquare := l.iWidth * l.iHeight
	wSquare := l.FWidth * l.FHeight
	wQube := l.iDepth * wSquare

	for fi := 0; fi < l.oDepth; fi++ {
		wi := fi * wQube
		for oy, initInputY := 0, -l.FPadding; oy < l.oHeight; oy, initInputY = oy+1, initInputY+l.FStride {
			for ox, initInputX := 0, -l.FPadding; ox < l.oWidth; ox, initInputX = ox+1, initInputX+l.FStride {

				l.output.Data[outXYZ] = l.Biases.Data[fi]

				for fy, iy := 0, initInputY; fy < l.FHeight; fy, iy = fy+1, iy+1 {
					for fx, ix := 0, initInputX; iy > -1 && iy < l.iHeight && fx < l.FWidth; fx, ix = fx+1, ix+1 {
						for iz := 0; ix > -1 && ix < l.iWidth && iz < l.iDepth; iz++ {
							inXYZ := iz*iSquare + iy*l.iWidth + ix
							wtXYZ := wi + iz*wSquare + fy*l.FWidth + fx

							l.output.Data[outXYZ] += l.Weights.Data[wtXYZ] * l.inputs.Data[inXYZ]
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
	weights := l.Weights.Copy()
	weights.RotateMatrixesInCube()

	outXYZ := 0
	iSquare := l.iWidth * l.iHeight
	wSquare := l.FWidth * l.FHeight
	wQube := l.iDepth * wSquare

	for fi := 0; fi < l.oDepth; fi++ {
		wi := fi * wQube
		for oy, initInputY := 0, -l.FPadding; oy < l.oHeight; oy, initInputY = oy+1, initInputY+l.FStride {
			for ox, initInputX := 0, -l.FPadding; ox < l.oWidth; ox, initInputX = ox+1, initInputX+l.FStride {

				l.gradBiases.Data[fi] += deltas.Data[outXYZ]

				for fy, iy := 0, initInputY; fy < l.FHeight; fy, iy = fy+1, iy+1 {
					for fx, ix := 0, initInputX; iy > -1 && iy < l.iHeight && fx < l.FWidth; fx, ix = fx+1, ix+1 {
						for iz := 0; ix > -1 && ix < l.iWidth && iz < l.iDepth; iz++ {
							inXYZ := iz*iSquare + iy*l.iWidth + ix
							wtXYZ := wi + iz*wSquare + fy*l.FWidth + fx
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

func (l *Conv) GetWeights() *nnet.Data {
	return l.Weights
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
	return l.Weights, l.gradWeights
}

func (l *Conv) GetBiasesWithGradient() (*nnet.Data, *nnet.Data) {
	return l.Biases, l.gradBiases
}
