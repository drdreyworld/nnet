package nnet

import (
	"errors"
	"fmt"
)

type LayerConstructor func() Layer

var LayersRegistry = layersRegistry{}

type layersRegistry map[string]LayerConstructor

func (reg layersRegistry) Create(key string) (Layer, error) {
	if constructor, ok := reg[key]; ok {
		return constructor(), nil
	}

	return nil, errors.New(fmt.Sprintf("Layer '%s' is not registered", key))
}

