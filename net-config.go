package nnet

import (
	"encoding/json"
	"errors"
)

const ERR_NETCONF_NOT_INITIALIZED = "Net config is not initialized"
const ERR_NETCONF_HAS_INVALID_TYPE = "Net config has invalid type"
const ERR_NETCONF_HAS_EMPTY_LAYERS = "Net config has empty layers"

type NetConfig struct {
	IWidth  int
	IHeight int
	IDepth  int

	OWidth  int
	OHeight int
	ODepth  int

	Layers   LayerConfigs
	Filename string
}

func CreateNetConfig(cfg interface{}) (nc NetConfig, err error) {
	if cfg == nil {
		err = errors.New(ERR_NETCONF_NOT_INITIALIZED)
		return
	}

	nc, ok := cfg.(NetConfig)
	if !ok {
		err = errors.New(ERR_NETCONF_HAS_INVALID_TYPE)
	}

	if err == nil && len(nc.Layers) == 0 {
		err = errors.New(ERR_NETCONF_HAS_EMPTY_LAYERS)
	}

	return
}

func (c *NetConfig) ToJSON() ([]byte, error) {
	return json.Marshal(c)
}
