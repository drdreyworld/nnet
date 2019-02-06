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

	log.Println("init layer: pooling, input sizes:", w, h, d, "output sizes:", l.oWidth, l.oHeight, l.oDepth)

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *Pooling) Activate(inputs *nnet.Data) *nnet.Data {
	l.inputs = inputs

	wW, wH := l.FWidth, l.FHeight

	outXYZ := 0
	iSquare := l.iWidth * l.iHeight

	max := 0.0
	for oz := 0; oz < l.oDepth; oz++ {
		for oy := 0; oy < l.oHeight; oy++ {
			for ox := 0; ox < l.oWidth; ox, outXYZ = ox+1, outXYZ+1 {

				for fy, iy := 0, oy*l.FStride-l.FPadding; fy < wH; fy, iy = fy+1, iy+1 {
					for fx, ix := 0, ox*l.FStride-l.FPadding; iy > -1 && iy < l.iHeight && fx < wW; fx, ix = fx+1, ix+1 {
						if ix > -1 && ix < l.iWidth {
							inXYZ := oz*iSquare + iy*l.iWidth + ix
							if (fy == 0 && fx == 0) || max < l.inputs.Data[inXYZ] {
								max = l.inputs.Data[inXYZ]
							}
						}
					}
				}

				l.output.Data[outXYZ] = max
			}
		}
	}
	return l.output
}

func (l *Pooling) Backprop(deltas *nnet.Data) (gradient *nnet.Data) {
	gradient = l.inputs.CopyZero()

	wW, wH := l.FWidth, l.FHeight

	outXYZ := 0
	iSquare := l.iWidth * l.iHeight

	for oz := 0; oz < l.oDepth; oz++ {
		for oy := 0; oy < l.oHeight; oy++ {
			for ox := 0; ox < l.oWidth; ox++ {
				max, giXYZ := 0.0, 0

				for fy, iy := 0, oy*l.FStride-l.FPadding; fy < wH; fy, iy = fy+1, iy+1 {
					for fx, ix := 0, ox*l.FStride-l.FPadding; iy > -1 && iy < l.iHeight && fx < wW; fx, ix = fx+1, ix+1 {
						if ix > -1 && ix < l.iWidth {
							inXYZ := oz*iSquare + iy*l.iWidth + ix
							if (fy == 0 && fx == 0) || max < l.inputs.Data[inXYZ] {
								max, giXYZ = l.inputs.Data[inXYZ], inXYZ
							}
						}
					}
				}

				gradient.Data[giXYZ] = deltas.Data[outXYZ]

				outXYZ++
			}
		}
	}

	return
}

func (l *Pooling) GetOutput() *nnet.Data {
	return l.output
}
