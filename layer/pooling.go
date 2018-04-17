package layer

import (
	"github.com/drdreyworld/nnet"
)

const LAYER_POOLING = "pooling"

func init() {
	nnet.LayersRegistry[LAYER_POOLING] = PoolingLayerConstructor
}

func PoolingLayerConstructor() nnet.Layer {
	return &Pooling{}
}

type Pooling struct {
	iWidth, iHeight, iDepth int
	oWidth, oHeight, oDepth int

	fwidth  int
	fheight int

	fstride  int
	fpadding int

	inputs *nnet.Data
	output *nnet.Data
}

func (l *Pooling) Init(config nnet.LayerConfig) (err error) {
	l.inputs = &nnet.Data{}
	l.output = &nnet.Data{}

	l.fwidth = config.Data.Int("FWidth")
	l.fheight = config.Data.Int("FHeight")
	l.fstride = config.Data.Int("Stride")
	l.fpadding = config.Data.Int("Padding")

	if l.fstride < 1 {
		l.fstride = 1
	}

	return
}

func (l *Pooling) InitDataSizes(w, h, d int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d

	l.oWidth = (l.iWidth-l.fwidth+2*l.fpadding)/l.fstride + 1
	l.oHeight = (l.iHeight-l.fheight+2*l.fpadding)/l.fstride + 1
	l.oDepth = l.iDepth

	l.output.InitCube(l.oWidth, l.oHeight, l.oDepth)

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *Pooling) Activate(inputs *nnet.Data) *nnet.Data {
	l.inputs = inputs

	wW, wH := l.fwidth, l.fheight

	outXYZ := 0
	iSquare := l.iWidth * l.iHeight

	max := 0.0
	for oz := 0; oz < l.oDepth; oz++ {
		for oy := 0; oy < l.oHeight; oy++ {
			for ox := 0; ox < l.oWidth; ox, outXYZ = ox+1, outXYZ+1 {

				for fy, iy := 0, oy*l.fstride-l.fpadding; fy < wH; fy, iy = fy+1, iy+1 {
					for fx, ix := 0, ox*l.fstride-l.fpadding; iy > -1 && iy < l.iHeight && fx < wW; fx, ix = fx+1, ix+1 {
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

	wW, wH := l.fwidth, l.fheight

	outXYZ := 0
	iSquare := l.iWidth * l.iHeight

	for oz := 0; oz < l.oDepth; oz++ {
		for oy := 0; oy < l.oHeight; oy++ {
			for ox := 0; ox < l.oWidth; ox++ {
				max, giXYZ := 0.0, 0

				for fy, iy := 0, oy*l.fstride-l.fpadding; fy < wH; fy, iy = fy+1, iy+1 {
					for fx, ix := 0, ox*l.fstride-l.fpadding; iy > -1 && iy < l.iHeight && fx < wW; fx, ix = fx+1, ix+1 {
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

func (l *Pooling) Serialize() (res nnet.LayerConfig) {
	res.Type = LAYER_POOLING
	res.Data = nnet.LayerConfigData{
		"FWidth":  l.fwidth,
		"FHeight": l.fheight,
		"Stride":  l.fstride,
		"Padding": l.fpadding,
	}
	return
}

func (l *Pooling) GetOutput() *nnet.Data {
	return l.output
}
