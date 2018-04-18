package nnet

import (
	"fmt"
)

type LayerConstructor func(cfg LayerConfig) (Layer, error)

var LayersRegistry = layersRegistry{}

type layersRegistry map[string]LayerConstructor

const ERR_LAYER_NOT_REGISTERED = "layer not registered with type %s"

func (reg layersRegistry) Create(cfg LayerConfig) (Layer, error) {
	if constructor, ok := reg[cfg.Type]; ok {
		return constructor(cfg)
	}

	return nil, fmt.Errorf(ERR_LAYER_NOT_REGISTERED, cfg.Type)
}

