package layer

import (
	"errors"
	"encoding/gob"
	"github.com/drdreyworld/nnet"
)

func init() {
	gob.Register(DenseConfig{})
}

type DenseConfig struct {
	// Sizes of output tensor
	OWidth, OHeight, ODepth int

	// Weights from previous learning
	Weights nnet.Mem

	// Biases from previous learning
	Biases nnet.Mem
}

func (cfg DenseConfig) Check() (err error) {
	opow := cfg.OWidth * cfg.OHeight * cfg.ODepth

	if opow <= 0 {
		return errors.New("Output sizes invalid")
	}

	return
}
