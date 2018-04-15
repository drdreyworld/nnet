package layer

import (
	"errors"
	"encoding/gob"
	"github.com/drdreyworld/nnet"
)

func init() {
	gob.Register(ConvConfig{})
}

type ConvConfig struct {
	// Width & Height of filter
	FWidth, FHeight int

	OutDepth int

	Padding int
	Stride  int

	// Convolution filter weights
	Weights nnet.Mem

	// Biases from previous learning
	Biases nnet.Mem
}

func (cfg ConvConfig) Check() (err error) {
	if cfg.FWidth*cfg.FHeight*cfg.OutDepth <= 0 {
		return errors.New("Filter sizes invalid")
	}

	return
}
