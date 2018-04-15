package layer

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"
	"github.com/drdreyworld/nnet"
)

const LAYER_CONV = "conv"

func init() {
	nnet.Layers[LAYER_CONV] = ConvLayerConstructor
}

func ConvLayerConstructor() nnet.Layer {
	return &Conv{}
}

type Conv struct {
	iWidth  int
	iHeight int
	iDepth  int

	oWidth  int
	oHeight int
	oDepth  int

	fwidth  int
	fheight int

	outDepth int

	fpadding int
	fstride  int

	MWeights *nnet.Mem
	MOutput  *nnet.Mem
	MInputs  *nnet.Mem
	MBiases  *nnet.Mem

	MGradWeights *nnet.Mem
	MGradBiases  *nnet.Mem
	MGradInputs  *nnet.Mem
}

func (l *Conv) Init(config nnet.LayerConfig) (err error) {

	l.MWeights = &nnet.Mem{}
	l.MOutput = &nnet.Mem{}
	l.MBiases = &nnet.Mem{}

	l.MGradWeights = &nnet.Mem{}
	l.MGradBiases = &nnet.Mem{}
	l.MGradInputs = &nnet.Mem{}

	if config.Data == nil {
		return errors.New("Config data is missed")
	}

	c, ok := config.Data.(ConvConfig)
	if !ok {
		return errors.New("Invalid config for Conv layer")
	}

	if err = c.Check(); err != nil {
		return
	}

	l.fwidth = c.FWidth
	l.fheight = c.FHeight

	l.outDepth = c.OutDepth

	l.fpadding = c.Padding
	l.fstride = c.Stride

	if l.fstride < 1 {
		l.fstride = 1
	}

	if len(c.Weights.Data) > 0 {
		l.MWeights = &c.Weights
		l.MBiases = &c.Biases
	}

	return
}

func (l *Conv) InitDataSizes(w, h, d int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d

	l.oWidth = (l.iWidth-l.fwidth+2*l.fpadding)/l.fstride + 1
	l.oHeight = (l.iHeight-l.fheight+2*l.fpadding)/l.fstride + 1
	l.oDepth = l.outDepth

	l.MOutput.InitTensor(l.oWidth, l.oHeight, l.oDepth)

	if len(l.MWeights.Data) == 0 {
		rand.Seed(time.Now().UnixNano())
		maxWeight := math.Sqrt(1.0 / float64(l.fwidth*l.fheight*l.outDepth*l.iDepth))

		l.MWeights.InitTensorRandom(l.fwidth, l.fheight, l.outDepth*l.iDepth, -maxWeight, maxWeight)
		l.MBiases.InitVector(l.outDepth)
	}

	l.MGradBiases.InitVector(l.outDepth)
	l.MGradWeights.InitTensor(l.fwidth, l.fheight, l.iDepth*l.outDepth)
	l.MGradInputs.InitTensor(l.iWidth, l.iHeight, l.iDepth)

	fmt.Println("conv inputs params:", l.iWidth, l.iHeight, l.iDepth)
	fmt.Println("conv output params:", l.oWidth, l.oHeight, l.oDepth)

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *Conv) Activate(inputs *nnet.Mem) *nnet.Mem {
	// inputs is readonly for layer
	l.MInputs = inputs

	outXYZ := 0
	iSquare := l.iWidth * l.iHeight
	wSquare := l.fwidth * l.fheight
	wQube := l.iDepth * wSquare

	for fi := 0; fi < l.oDepth; fi++ {
		wi := fi * wQube
		for oy, initInputY := 0, -l.fpadding; oy < l.oHeight; oy, initInputY = oy+1, initInputY+l.fstride {
			for ox, initInputX := 0, -l.fpadding; ox < l.oWidth; ox, initInputX = ox+1, initInputX+l.fstride {

				l.MOutput.Data[outXYZ] = l.MBiases.Data[fi]

				for fy, iy := 0, initInputY; fy < l.fheight; fy, iy = fy+1, iy+1 {
					for fx, ix := 0, initInputX; iy > -1 && iy < l.iHeight && fx < l.fwidth; fx, ix = fx+1, ix+1 {
						for iz := 0; ix > -1 && ix < l.iWidth && iz < l.iDepth; iz++ {
							inXYZ := iz*iSquare + iy*l.iWidth + ix
							wtXYZ := wi + iz*wSquare + fy*l.fwidth + fx

							l.MOutput.Data[outXYZ] += l.MWeights.Data[wtXYZ] * l.MInputs.Data[inXYZ]
						}
					}
				}

				outXYZ++
			}
		}
	}

	// output is readonly for next layer
	return l.MOutput
}

func (l *Conv) Backprop(deltas *nnet.Mem) *nnet.Mem {
	weights := l.MWeights.RotateTensorMatrixes()

	outXYZ := 0
	iSquare := l.iWidth * l.iHeight
	wSquare := l.fwidth * l.fheight
	wQube := l.iDepth * wSquare

	for fi := 0; fi < l.oDepth; fi++ {
		wi := fi * wQube
		for oy, initInputY := 0, -l.fpadding; oy < l.oHeight; oy, initInputY = oy+1, initInputY+l.fstride {
			for ox, initInputX := 0, -l.fpadding; ox < l.oWidth; ox, initInputX = ox+1, initInputX+l.fstride {

				l.MGradBiases.Data[fi] += deltas.Data[outXYZ]

				for fy, iy := 0, initInputY; fy < l.fheight; fy, iy = fy+1, iy+1 {
					for fx, ix := 0, initInputX; iy > -1 && iy < l.iHeight && fx < l.fwidth; fx, ix = fx+1, ix+1 {
						for iz := 0; ix > -1 && ix < l.iWidth && iz < l.iDepth; iz++ {
							inXYZ := iz*iSquare + iy*l.iWidth + ix
							wtXYZ := wi + iz*wSquare + fy*l.fwidth + fx
							l.MGradInputs.Data[inXYZ] += weights.Data[wtXYZ] * deltas.Data[outXYZ]
							l.MGradWeights.Data[wtXYZ] += l.MInputs.Data[inXYZ] * deltas.Data[outXYZ]
						}
					}
				}

				outXYZ++
			}
		}
	}

	return l.MGradInputs
}

func (l *Conv) Serialize() (res nnet.LayerConfig) {
	res.Type = LAYER_CONV
	res.Data = ConvConfig{
		FWidth:  l.fwidth,
		FHeight: l.fheight,

		OutDepth: l.outDepth,

		Padding: l.fpadding,
		Stride:  l.fstride,

		Weights: *l.MWeights,
		Biases:  *l.MBiases,
	}
	return
}

func (l *Conv) UnmarshalConfigDataFromJSON(b []byte) (interface{}, error) {
	cfg := ConvConfig{}
	err := json.Unmarshal(b, &cfg)

	return cfg, err
}

func (l *Conv) GetWeights() *nnet.Mem {
	return l.MWeights
}

func (l *Conv) GetOutput() *nnet.Mem {
	return l.MOutput
}

func (l *Conv) ResetGradients() {
	l.MGradBiases.Fill(0)
	l.MGradWeights.Fill(0)
	l.MGradInputs.Fill(0)
}

func (l *Conv) GetWeightsWithGradient() (*nnet.Mem, *nnet.Mem) {
	return l.MWeights, l.MGradWeights
}

func (l *Conv) GetBiasesWithGradient() (*nnet.Mem, *nnet.Mem) {
	return l.MBiases, l.MGradBiases
}
