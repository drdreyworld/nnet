package layer

import (
	"encoding/gob"
	"github.com/drdreyworld/nnet"
	"log"
	"sync"
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
	FWidth, FHeight, FDepth int
	oWidth, oHeight, oDepth int

	FCount   int
	FPadding int
	FStride  int

	Weights *nnet.Data
	Biases  *nnet.Data

	inputs *nnet.Data
	output *nnet.Data

	gradWeights *nnet.Data
	gradBiases  *nnet.Data
	gradInputs  *nnet.Data

	iSquare int
	oSquare int
	wSquare int
	wCube   int

	FMutable bool
}

func (l *Conv) GetType() string {
	return LAYER_CONV
}

func (l *Conv) InitDataSizes(iw, ih, id int) (int, int, int) {
	if l.FStride == 0 {
		l.FStride = 1
	}

	l.iWidth, l.iHeight, l.iDepth = iw, ih, id
	l.oWidth, l.oHeight, l.oDepth = (iw-l.FWidth+2*l.FPadding)/l.FStride+1, (ih-l.FHeight+2*l.FPadding)/l.FStride+1, l.FCount
	l.FDepth = id

	if l.Weights == nil {
		l.Weights = &nnet.Data{}
		l.Biases = &nnet.Data{}
	}

	if len(l.Weights.Data) == 0 {
		//maxWeight := math.Sqrt(1.0 / float64(l.FWidth*l.FHeight*l.FCount*l.FDepth))
		l.Weights.InitCubeRandom(l.FWidth, l.FHeight, l.FCount*l.FDepth, -0.7, 0.7)
		//l.Weights.InitCubeRandom(l.FWidth, l.FHeight, l.FCount*l.FDepth, -1, 1)
		l.Biases.InitVector(l.FCount)
		l.Biases.Fill(0.1) // for relu
	}

	l.output = &nnet.Data{}
	l.output.InitCube(l.oWidth, l.oHeight, l.oDepth)

	l.gradBiases = &nnet.Data{}
	l.gradBiases.InitVector(l.FCount)

	l.gradWeights = &nnet.Data{}
	l.gradWeights.InitCube(l.FWidth, l.FHeight, l.FCount*l.FDepth)

	l.gradInputs = &nnet.Data{}
	l.gradInputs.InitCube(l.iWidth, l.iHeight, l.iDepth)

	l.iSquare = l.iWidth * l.iHeight
	l.oSquare = l.oWidth * l.oHeight
	l.wSquare = l.FWidth * l.FHeight
	l.wCube = l.FDepth * l.wSquare

	log.Println("init layer: conv,",
		"input sizes:", iw, ih, id,
		"output sizes:", l.oWidth, l.oHeight, l.oDepth,
		"matrix sizes:", l.FWidth, "x", l.FHeight, "x", l.FDepth,
		"count:", l.FCount,
	)

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *Conv) Activate(inputs *nnet.Data) *nnet.Data {
	l.inputs = inputs

	wg := new(sync.WaitGroup)
	wg.Add(l.oDepth)

	for filterIndex := 0; filterIndex < l.FCount; filterIndex++ {
		go l.activateFilter(wg, filterIndex)
	}
	wg.Wait()

	return l.output
}

func (l *Conv) activateFilter(wg *sync.WaitGroup, fi int) {
	defer wg.Done()

	outXYZ := fi * l.oSquare

	wi := fi * l.wCube
	for oy, initInputY := 0, -l.FPadding; oy < l.oHeight; oy, initInputY = oy+1, initInputY+l.FStride {
		for ox, initInputX := 0, -l.FPadding; ox < l.oWidth; ox, initInputX = ox+1, initInputX+l.FStride {

			l.output.Data[outXYZ] = l.Biases.Data[fi]

			for fy, iy := 0, initInputY; fy < l.FHeight; fy, iy = fy+1, iy+1 {
				for fx, ix := 0, initInputX; iy > -1 && iy < l.iHeight && fx < l.FWidth; fx, ix = fx+1, ix+1 {
					for iz := 0; ix > -1 && ix < l.iWidth && iz < l.iDepth; iz++ {
						inXYZ := iz*l.iSquare + iy*l.iWidth + ix
						wtXYZ := wi + iz*l.wSquare + fy*l.FWidth + fx

						l.output.Data[outXYZ] += l.Weights.Data[wtXYZ] * l.inputs.Data[inXYZ]
					}
				}
			}

			outXYZ++
		}
	}
}

func (l *Conv) Backprop(deltas *nnet.Data) *nnet.Data {
	//weights := l.Weights.Copy()
	//weights.RotateMatrixesInCube()

	wg := new(sync.WaitGroup)
	wg.Add(l.oDepth)

	for fi := 0; fi < l.oDepth; fi++ {
		go l.backpropFilter(wg, fi, deltas, l.Weights)
	}
	wg.Wait()

	return l.gradInputs
}

func (l *Conv) backpropFilter(wg *sync.WaitGroup, fi int, deltas *nnet.Data, weights *nnet.Data) {
	defer wg.Done()

	outXYZ := fi * l.oSquare

	wi := fi * l.wCube

	for oy, initInputY := 0, -l.FPadding; oy < l.oHeight; oy, initInputY = oy+1, initInputY+l.FStride {
		for ox, initInputX := 0, -l.FPadding; ox < l.oWidth; ox, initInputX = ox+1, initInputX+l.FStride {

			l.gradBiases.Data[fi] += deltas.Data[outXYZ]

			for fy, iy := 0, initInputY; fy < l.FHeight; fy, iy = fy+1, iy+1 {
				for fx, ix := 0, initInputX; iy > -1 && iy < l.iHeight && fx < l.FWidth; fx, ix = fx+1, ix+1 {
					for iz := 0; ix > -1 && ix < l.iWidth && iz < l.iDepth; iz++ {
						inXYZ := iz*l.iSquare + iy*l.iWidth + ix
						wtXYZ := wi + iz*l.wSquare + fy*l.FWidth + fx
						l.gradInputs.Data[inXYZ] += weights.Data[wtXYZ] * deltas.Data[outXYZ]
						l.gradWeights.Data[wtXYZ] += l.inputs.Data[inXYZ] * deltas.Data[outXYZ]
					}
				}
			}

			outXYZ++
		}
	}
}

func (l *Conv) GetWeights() *nnet.Data {
	return l.Weights
}

func (l *Conv) GetOutput() *nnet.Data {
	return l.output
}

func (l *Conv) GetInputs() *nnet.Data {
	return l.inputs
}

func (l *Conv) ResetGradients() {
	l.gradInputs.Reset()
	l.gradBiases.Reset()
	l.gradWeights.Reset()
}

func (l *Conv) GetWeightsWithGradient() (*nnet.Data, *nnet.Data) {
	return l.Weights, l.gradWeights
}

func (l *Conv) GetBiasesWithGradient() (*nnet.Data, *nnet.Data) {
	return l.Biases, l.gradBiases
}

func (l *Conv) GetInputGradients() (g *nnet.Data) {
	return l.gradInputs
}

func (l *Conv) Mutable() bool {
	return l.FMutable
}
