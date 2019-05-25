package trainer

import (
	"github.com/drdreyworld/nnet"
)

type VanillaSGDExt struct {
	Network nnet.Net

	Learning    float64
	Momentum    float64
	WeightDecay float64

	Output    *nnet.Data
	Deltas    *nnet.Data
	Gradients []*nnet.Data
}

func (t *VanillaSGDExt) SetNet(n nnet.Net) {
	t.Network = n
}

func (t *VanillaSGDExt) InitGradients() {
	t.Gradients = []*nnet.Data{}
	for i := 0; i < t.Network.GetLayersCount(); i++ {
		if layer, ok := t.Network.GetLayer(i).(nnet.TrainableLayer); ok {
			_, g := layer.GetWeightsWithGradient()
			t.Gradients = append(t.Gradients, g.CopyZero())

			_, g = layer.GetBiasesWithGradient()
			t.Gradients = append(t.Gradients, g.CopyZero())
		}
	}
}

func (t *VanillaSGDExt) Activate(inputs, target *nnet.Data) *nnet.Data {
	t.Output = t.Network.Activate(inputs)
	t.Deltas = t.Network.GetOutputDeltas(target, t.Output)

	t.Network.Backprop(t.Deltas)

	return t.Output
}

func (t *VanillaSGDExt) UpdateWeights() {
	if len(t.Gradients) == 0 {
		t.InitGradients()
	}

	k := 0
	for i := 0; i < t.Network.GetLayersCount(); i++ {
		layer, ok := t.Network.GetLayer(i).(nnet.TrainableLayer)
		if ok {
			if !layer.Mutable() {
				k += 2
				layer.ResetGradients()
				continue
			}

			w, g := layer.GetWeightsWithGradient()
			for j := 0; j < len(w.Data); j++ {
				value := t.Gradients[k].Data[j]*t.Momentum + t.Learning*g.Data[j] + t.WeightDecay*w.Data[j]

				w.Data[j] -= value
				t.Gradients[k].Data[j] = value
			}
			k++

			w, g = layer.GetBiasesWithGradient()
			for j := 0; j < len(w.Data); j++ {
				value := t.Gradients[k].Data[j]*t.Momentum + t.Learning*g.Data[j] + t.WeightDecay*w.Data[j]

				w.Data[j] -= value
				t.Gradients[k].Data[j] = value
			}
			k++

			layer.ResetGradients()
		}
	}
}
