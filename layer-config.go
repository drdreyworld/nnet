package nnet

type LayerConfig struct {
	Type string
	Data LayerConfigData
}

type LayerConfigData map[string]interface{}

func (d LayerConfigData) Exists(key string) bool {
	_, ok := d[key]
	return ok
}

func (d LayerConfigData) Int(key string) int {
	// after json unmarshaling all nums is float64 :/
	if v, ok := d[key].(float64); ok {
		return int(v)
	}

	return d[key].(int)
}

func (d LayerConfigData) Float64(key string) float64 {
	return d[key].(float64)
}

func (d LayerConfigData) String(key string) string {
	return d[key].(string)
}