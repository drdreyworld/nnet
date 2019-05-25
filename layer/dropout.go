package layer

import (
	"encoding/gob"
	"github.com/drdreyworld/nnet"
	"log"
	"math/rand"
)

const LAYER_DROPUOT = "dropuot"

func init() {
	nnet.LayersRegistry[LAYER_DROPUOT] = LayerConstructorDropOut
	gob.Register(DropOut{})
}

func LayerConstructorDropOut() nnet.Layer {
	return &DropOut{}
}

type DropOut struct {
	iWidth, iHeight, iDepth int
	oWidth, oHeight, oDepth int

	FProbability float64

	droppped []bool
	randbuff []byte
	randbyte byte

	output *nnet.Data
}

func (l *DropOut) GetType() string {
	return LAYER_DROPUOT
}

func (l *DropOut) InitDataSizes(w, h, d int) (int, int, int) {
	l.iWidth, l.iHeight, l.iDepth = w, h, d
	l.oWidth, l.oHeight, l.oDepth = w, h, d

	log.Println("init layer: dropuot, input sizes:", w, h, d, "output sizes:", l.oWidth, l.oHeight, l.oDepth)

	l.output = new(nnet.Data)
	l.output.InitCube(w, h, d)

	l.droppped = make([]bool, w*h*d)
	l.randbuff = make([]byte, w*h*d)
	l.randbyte = byte(255 * l.FProbability)

	return l.oWidth, l.oHeight, l.oDepth
}

func (l *DropOut) Activate(inputs *nnet.Data) *nnet.Data {
	l.output = inputs.Copy()

	_, err := rand.Read(l.randbuff)
	if err != nil {
		log.Fatalln(err)
	}

	for i := 0; i < len(inputs.Data); i++ {
		if l.droppped[i] = l.randbuff[i] <= l.randbyte; l.droppped[i] {
			l.output.Data[i] = 0
		}
	}

	return l.output
}

func (l *DropOut) Backprop(deltas *nnet.Data) (gradient *nnet.Data) {
	gradient = deltas.Copy()

	for i := 0; i < len(deltas.Data); i++ {
		if l.droppped[i] {
			gradient.Data[i] = 0
		}
	}
	return
}

func (l *DropOut) GetOutput() *nnet.Data {
	return l.output
}
