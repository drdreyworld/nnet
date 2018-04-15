package nnet

import (
	"encoding/json"
)

var Configs = map[string]interface{}{}

type LayerConfigs []LayerConfig

type LayerConfig struct {
	Type string
	Data interface{}
}

func (c *LayerConfig) UnmarshalJSON(b []byte) (err error) {
	cfg := struct {
		Type string
		Data *json.RawMessage
	}{}

	if err = json.Unmarshal(b, &cfg); err != nil {
		return err
	}

	l, err := Layers.Create(cfg.Type)
	if err != nil {
		return err
	}

	c.Type = cfg.Type

	if cfg.Data != nil {
		c.Data, err = l.UnmarshalConfigDataFromJSON(*cfg.Data)
	}

	return
}
