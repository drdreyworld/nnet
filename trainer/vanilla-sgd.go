package trainer

import "github.com/drdreyworld/nnet"

type VanilaSGD struct {
	Network  nnet.Net
	Learning float64
	Output   *nnet.Data
	Deltas   *nnet.Data
}

func (t *VanilaSGD) SetNet(n nnet.Net) {
	t.Network = n
}

func (t *VanilaSGD) Activate(inputs, target *nnet.Data) *nnet.Data {
	t.Output = t.Network.Activate(inputs)
	t.Deltas = t.Network.GetOutputDeltas(target, t.Output)

	t.Network.Backprop(t.Deltas)

	return t.Output
}

func (t *VanilaSGD) UpdateWeights() {
	for i := 0; i < t.Network.GetLayersCount(); i++ {
		layer, ok := t.Network.GetLayer(i).(nnet.TrainableLayer)
		if ok && layer.Mutable() {
			w, g := layer.GetWeightsWithGradient()
			for i := 0; i < len(w.Data); i++ {
				w.Data[i] -= t.Learning * g.Data[i]
			}

			w, g = layer.GetBiasesWithGradient()
			for i := 0; i < len(w.Data); i++ {
				w.Data[i] -= t.Learning * g.Data[i]
			}

			layer.ResetGradients()
		}
	}
}
