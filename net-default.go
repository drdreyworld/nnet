package nnet

import (
	"errors"
)

const ERR_NET_NOT_SET_LOSS = "loss function not set"

type NetDefault struct {
	IWidth, IHeight, IDepth int
	OWidth, OHeight, ODepth int

	LossFunc LossFunction
	LossCode string

	layers   []Layer
}

func (n *NetDefault) Init(cfg NetConfig) (err error) {
	n.layers, err = cfg.CreateLayers()
	if err != nil {
		return
	}

	n.LossCode = cfg.LossCode
	n.LossFunc = LossRegistry[n.LossCode]

	n.IWidth, n.IHeight, n.IDepth = cfg.IWidth, cfg.IHeight, cfg.IDepth
	n.OWidth, n.OHeight, n.ODepth = cfg.OWidth, cfg.OHeight, cfg.ODepth

	return
}

func (n *NetDefault) Activate(inputs *Data) *Data {
	for i := 0; i < len(n.layers); i++ {
		inputs = n.layers[i].Activate(inputs)
	}
	return inputs
}

func (n *NetDefault) Backprop(deltas *Data) (gradient *Data) {
	gradient = deltas.Copy()

	for i := len(n.layers) - 1; i >= 0; i-- {
		gradient = n.layers[i].Backprop(gradient)
	}
	return gradient
}

func (n *NetDefault) GetOutputDeltas(target, output *Data) (res *Data) {
	res = target.CopyZero()
	for i := 0; i < len(res.Data); i++ {
		res.Data[i] = -(target.Data[i] - output.Data[i])
	}
	return
}

func (n *NetDefault) Serialize() NetConfig {
	res := NetConfig{}

	res.IWidth, res.IHeight, res.IDepth = n.IWidth, n.IHeight, n.IDepth
	res.OWidth, res.OHeight, res.ODepth = n.OWidth, n.OHeight, n.ODepth

	res.LossCode = n.LossCode

	for i := 0; i < len(n.layers); i++ {
		res.Layers = append(res.Layers, n.layers[i].Serialize())
	}
	return res
}

func (n *NetDefault) GetLoss(target, output *Data) (float64, error) {
	if n.LossFunc == nil {
		return 0, errors.New(ERR_NET_NOT_SET_LOSS)
	}
	return n.LossFunc(target, output), nil
}

func (n *NetDefault) GetLayersCount() int {
	return len(n.layers)
}

func (n *NetDefault) GetLayer(index int) Layer {
	if index > -1 && index < len(n.layers) {
		return n.layers[index]
	}
	return nil
}
