package nnet

type LayerConfigData map[string]interface{}

const (
	KEY_ACTIVATION = "Activation"
	KEY_WEIGHTS    = "Weights"
	KEY_BIASES     = "Biases"
)

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

func (d LayerConfigData) GetActivation() Activation {
	aCode := d.String(KEY_ACTIVATION)
	return ActivationsRegistry[aCode]
}

func (d LayerConfigData) SetActivation(afunc Activation) {
	d[KEY_ACTIVATION] = afunc.Serialize()
}

func (d LayerConfigData) GetWeights() *Data {
	if weights, ok := d[KEY_WEIGHTS].(Data); ok {
		return &weights
	}
	return &Data{}
}

func (d LayerConfigData) SetWeights(weights *Data) {
	if weights != nil {
		d[KEY_WEIGHTS] = *weights
	}
}

func (d LayerConfigData) GetBiases() *Data {
	if biases, ok := d[KEY_BIASES].(Data); ok {
		return &biases
	}
	return &Data{}
}

func (d LayerConfigData) SetBiases(biases *Data) {
	if biases != nil {
		d[KEY_BIASES] = *biases
	}
}
