package nnet

import (
	"fmt"
)

type LayerConstructor func() Layer

var LayersRegistry = layersRegistry{}

type layersRegistry map[string]LayerConstructor

const ERR_LAYER_NOT_REGISTERED = "layer not registered with type %s"

func (reg layersRegistry) Create(LayerType string) (Layer, error) {
	if constructor, ok := reg[LayerType]; ok {
		return constructor(), nil
	}

	return nil, fmt.Errorf(ERR_LAYER_NOT_REGISTERED, LayerType)
}

