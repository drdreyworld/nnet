package layer

import (
	"errors"
	"encoding/gob"
)

func init() {
	gob.Register(PoolingConfig{})
}

type PoolingConfig struct {
	FWidth  int
	FHeight int
	Stride  int
	Padding int
}

func (cfg PoolingConfig) Check() (err error) {
	if cfg.FWidth <= 0 {
		return errors.New("width must be greater than zero")
	}

	if cfg.FHeight <= 0 {
		return errors.New("height must be greater than zero")
	}

	if cfg.Stride <= 0 {
		return errors.New("fstride must be greater than zero")
	}

	return
}
