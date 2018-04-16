package nnet

import (
	"errors"
)

type NNet interface {
	Init(cfg NetConfig) error
	Activate(inputs *Data) (output *Data)
	Backprop(deltas *Data) (gradient *Data)
	GetOutputDeltas(target, output *Data) (res *Data)
	GetLayersCount() int
	GetLayer(index int) Layer

	Serialize() NetConfig
}

const ERR_NET_NOT_SET_LOSS = "loss function not set"
const ERR_NET_NOT_SET_STORAGE = "storage instance not set"

type Net struct {
	IWidth, IHeight, IDepth int
	OWidth, OHeight, ODepth int

	LossFunc LossFunction
	LossCode string

	layers   []Layer
	Storage  NetStorage
}

func (n *Net) Init(cfg NetConfig) (err error) {
	n.layers, err = cfg.CreateLayers()
	if err != nil {
		return
	}

	n.LossCode = cfg.LossCode
	n.LossFunc = LossRegistry[n.LossCode]

	n.IWidth, n.IHeight, n.IDepth = cfg.IWidth, cfg.IHeight, cfg.IDepth
	n.OWidth, n.OHeight, n.ODepth = cfg.IWidth, cfg.IHeight, cfg.IDepth

	for i := 0; i < len(n.layers); i++ {
		n.OWidth, n.OHeight, n.ODepth = n.layers[i].InitDataSizes(n.OWidth, n.OHeight, n.ODepth)
	}

	return
}

func (n *Net) Activate(inputs *Data) *Data {
	for i := 0; i < len(n.layers); i++ {
		inputs = n.layers[i].Activate(inputs)
	}
	return inputs
}

func (n *Net) Backprop(deltas *Data) (gradient *Data) {
	gradient = deltas.Copy()

	for i := len(n.layers) - 1; i >= 0; i-- {
		gradient = n.layers[i].Backprop(gradient)
	}
	return gradient
}

func (n *Net) GetOutputDeltas(target, output *Data) (res *Data) {
	res = target.CopyZero()
	for i := 0; i < len(res.Data); i++ {
		res.Data[i] = -(target.Data[i] - output.Data[i])
	}
	return
}

func (n *Net) Serialize() NetConfig {
	res := NetConfig{}

	res.IWidth, res.IHeight, res.IDepth = n.IWidth, n.IHeight, n.IDepth
	res.OWidth, res.OHeight, res.ODepth = n.OWidth, n.OHeight, n.ODepth

	res.LossCode = n.LossCode

	for i := 0; i < len(n.layers); i++ {
		res.Layers = append(res.Layers, n.layers[i].Serialize())
	}
	return res
}

func (n *Net) GetLoss(target, output *Data) (float64, error) {
	if n.LossFunc == nil {
		return 0, errors.New(ERR_NET_NOT_SET_LOSS)
	}
	return n.LossFunc(target, output), nil
}

func (n *Net) Save() error {
	if n.Storage != nil {
		return n.Storage.Save(n)
	}
	return errors.New(ERR_NET_NOT_SET_STORAGE)
}

func (n *Net) Load() error {
	if n.Storage != nil {
		return n.Storage.Load(n)
	}
	return errors.New(ERR_NET_NOT_SET_STORAGE)
}

func (n *Net) GetLayersCount() int {
	return len(n.layers)
}

func (n *Net) GetLayer(index int) Layer {
	if index > -1 && index < len(n.layers) {
		return n.layers[index]
	}
	return nil
}
