package layer

import (
	"github.com/drdreyworld/nnet"
	"testing"
)

func TestDense_Complex(t *testing.T) {
	l := LayerConstructorDense().(*Dense)
	l.OWidth = 1
	l.OHeight = 2
	l.ODepth = 3


	iw, ih, id := 2, 2, 2
	ow, oh, od := l.InitDataSizes(iw, ih, id)

	if ow != 1 || oh != 2 || od != 3 {
		t.Error("output sizes not equal input sizes")
	}

	inputs := &nnet.Data{}
	inputs.InitCube(iw, ih, id)
	inputs.Data = []float64{
		10, 11,
		12, 13,

		14, 15,
		16, 17,
	}

	output := l.Activate(inputs)

	if l.GetOutput() != output {
		t.Error("getOutput returns other result than activate")
	}

	if output.Dims[0] != ow || output.Dims[1] != oh || output.Dims[2] != od {
		t.Error("output dimentions mistmatch")
	}

	// @todo test calc output algorithm
}
