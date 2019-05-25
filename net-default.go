package nnet

import (
	"encoding/gob"
	"errors"
)

func init() {
	gob.Register(NetDefault{})
}

type NetDefault struct {
	IWidth, IHeight, IDepth int
	OWidth, OHeight, ODepth int

	Loss string       // user defined loss code
	loss LossFunction // initiaized function by code

	Layers Layers
}

func (n *NetDefault) Init() (err error) {
	if f, ok := LossRegistry[n.Loss]; !ok {
		return errors.New("loss not registered: " + n.Loss)
	} else {
		n.loss = f
	}

	w, h, d := n.IWidth, n.IHeight, n.IDepth

	for i := 0; i < len(n.Layers); i++ {
		w, h, d = n.Layers[i].InitDataSizes(w, h, d)
	}

	n.OWidth, n.OHeight, n.ODepth = w, h, d

	return
}

func (n *NetDefault) Activate(inputs *Data) *Data {
	for i := 0; i < len(n.Layers); i++ {
		inputs = n.Layers[i].Activate(inputs)
	}
	return inputs
}

func (n *NetDefault) Backprop(deltas *Data) (gradient *Data) {
	gradient = deltas.Copy()

	for i := len(n.Layers) - 1; i >= 0; i-- {
		gradient = n.Layers[i].Backprop(gradient)
	}
	return gradient
}

func (n *NetDefault) GetOutputDeltas(target, output *Data) (res *Data) {
	res = output.CopyZero()
	for i := 0; i < len(target.Data); i++ {
		res.Data[i] = -(target.Data[i] - output.Data[i])
	}
	return
}

func (n *NetDefault) GetLoss(target, output *Data) float64 {
	return n.loss(target, output)
}

func (n *NetDefault) GetLayersCount() int {
	return len(n.Layers)
}

func (n *NetDefault) GetLayer(index int) Layer {
	if index > -1 && index < len(n.Layers) {
		return n.Layers[index]
	}
	return nil
}
