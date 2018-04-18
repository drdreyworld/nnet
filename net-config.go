package nnet

type NetConfig struct {
	IWidth, IHeight, IDepth int
	OWidth, OHeight, ODepth int

	LossCode string

	Layers []LayerConfig
}

func (c *NetConfig) CreateLayers() (res []Layer, err error) {
	var l Layer
	var w, h, d = c.IWidth, c.IHeight, c.IDepth

	for _, v := range c.Layers {

		l, err = LayersRegistry.Create(v)
		if err != nil {
			return nil, err
		}

		w, h, d = l.InitDataSizes(w, h, d)

		res = append(res, l)
	}
	return
}
