package layer

import (
	"encoding/gob"
	"github.com/drdreyworld/nnet"
	"log"
)

const LAYER_POOLING = "pooling"

func init() {
	nnet.LayersRegistry[LAYER_POOLING] = LayerConstructorPooling
	gob.Register(Pooling{})
}

func LayerConstructorPooling() nnet.Layer {
	return &Pooling{}
}

type Pooling struct {
	iWidth, iHeight, iDepth int
	oWidth, oHeight, oDepth int

	FWidth  int
	FHeight int

	FStride  int
	FPadding int

	inputs *nnet.Data
	output *nnet.Data
	coords []int
}

func (l *Pooling) GetType() string {
	return LAYER_POOLING
}

func (l *Pooling) InitDataSizes(w, h, d int) (int, int, int) {
	if l.FStride < 1 {
		l.FStride = 1
	}

	l.iWidth, l.iHeight, l.iDepth = w, h, d

	l.oWidth = (l.iWidth-l.FWidth+2*l.FPadding)/l.FStride + 1
	l.oHeight = (l.iHeight-l.FHeight+2*l.FPadding)/l.FStride + 1
	l.oDepth = l.iDepth

	l.output = &nnet.Data{}
	l.output.InitCube(l.oWidth, l.oHeight, l.oDepth)

	l.coords = make([]int, l.oWidth*l.oHeight*l.oDepth)

	log.Println("init layer: pooling, input sizes:", w, h, d, "output sizes:", l.oWidth, l.oHeight, l.oDepth)

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *Pooling) Activate(inputs *nnet.Data) *nnet.Data {
	l.inputs = inputs
	l.output.Fill(0)
	l.coords = make([]int, l.oWidth*l.oHeight*l.oDepth)

	wW, wH := l.FWidth, l.FHeight

	outXYZ := 0
	iSquare := l.iWidth * l.iHeight

	max := 0.0
	maxCoord := 0
	for oz := 0; oz < l.oDepth; oz++ {
		for oy := 0; oy < l.oHeight; oy++ {
			for ox := 0; ox < l.oWidth; ox++ {

				iy, n := oy*l.FStride-l.FPadding, true

				for fy := 0; fy < wH; fy++ {
					ix := ox*l.FStride - l.FPadding
					for fx := 0; fx < wW; fx++ {
						if ix > -1 && ix < l.iWidth && iy > -1 && iy < l.iHeight {
							inXYZ := oz*iSquare + iy*l.iWidth + ix

							if n || max < l.inputs.Data[inXYZ] {
								max, maxCoord, n = l.inputs.Data[inXYZ], inXYZ, false
							}
						}

						ix++
					}
					iy++
				}

				l.output.Data[outXYZ] = max
				l.coords[outXYZ] = maxCoord

				outXYZ++
			}
		}
	}
	return l.output
}

func (l *Pooling) Backprop(deltas *nnet.Data) (gradient *nnet.Data) {
	gradient = l.inputs.CopyZero()

	for i := 0; i < len(deltas.Data); i++ {
		gradient.Data[l.coords[i]] += deltas.Data[i]
	}
	return
}

func (l *Pooling) GetOutput() *nnet.Data {
	return l.output
}
