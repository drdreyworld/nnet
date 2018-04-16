package nnet

import (
	"errors"
	"fmt"
)

var Layers = LayerRegistry{}

type LayerConstructor func() Layer

type LayerRegistry map[string]LayerConstructor

func (reg LayerRegistry) Create(key string) (Layer, error) {
	if constructor, ok := reg[key]; ok {
		return constructor(), nil
	}

	return nil, errors.New(fmt.Sprintf("Layer '%s' is not registered", key))
}

type Layer interface {
	Init(cfg LayerConfig) (err error)
	InitDataSizes(w, h, d int) (int, int, int)
	Activate(inputs *Data) (output *Data)
	Backprop(deltas *Data) (nextDeltas *Data)
	Serialize() LayerConfig
	GetOutput() *Data
}

type TrainableLayer interface {
	ResetGradients()
	GetWeightsWithGradient() (w, g *Data)
	GetBiasesWithGradient() (w, g *Data)
}