package nnet

import "errors"

const (
	ERR_INVALID_LAYER_TYPE = "invalid layer type"
)

type LayerConfig struct {
	Type string
	Data LayerConfigData
}

func (c LayerConfig) CheckType(need string) error {
	if need != c.Type {
		return errors.New(ERR_INVALID_LAYER_TYPE)
	}
	return nil
}