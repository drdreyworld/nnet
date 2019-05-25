package trainer

import (
	"github.com/drdreyworld/nnet"
)

type VanillaSGDExtBatch struct {
	Network nnet.Net

	Learning    float64
	Momentum    float64
	WeightDecay float64
	BatchSize   int
	BatchIndex  int

	Output    *nnet.Data
	Deltas    *nnet.Data
	Gradients []*nnet.Data
	SumGrads  []*nnet.Data
}

func (t *VanillaSGDExtBatch) SetNet(n nnet.Net) {
	t.Network = n
}

func (t *VanillaSGDExtBatch) InitGradients() {
	t.Gradients = []*nnet.Data{}
	t.SumGrads = []*nnet.Data{}
	for i := 0; i < t.Network.GetLayersCount(); i++ {
		if layer, ok := t.Network.GetLayer(i).(nnet.TrainableLayer); ok {
			_, g := layer.GetWeightsWithGradient()
			t.Gradients = append(t.Gradients, g.CopyZero())
			t.SumGrads = append(t.SumGrads, g.CopyZero())

			_, g = layer.GetBiasesWithGradient()
			t.Gradients = append(t.Gradients, g.CopyZero())
			t.SumGrads = append(t.SumGrads, g.CopyZero())
		}
	}
}

func (t *VanillaSGDExtBatch) Activate(inputs, target *nnet.Data) *nnet.Data {
	t.Output = t.Network.Activate(inputs)
	t.Deltas = t.Network.GetOutputDeltas(target, t.Output)

	t.Network.Backprop(t.Deltas)

	if len(t.Gradients) == 0 {
		t.InitGradients()
	}

	if t.BatchSize < 1 {
		t.BatchSize = 1
	}

	t.BatchIndex++

	k := 0
	for i := 0; i < t.Network.GetLayersCount(); i++ {
		layer, ok := t.Network.GetLayer(i).(nnet.TrainableLayer)
		if ok {
			if !layer.Mutable() {
				k += 2
				layer.ResetGradients()
				continue
			}

			_, g := layer.GetWeightsWithGradient()
			for j := 0; j < len(g.Data); j++ {
				t.SumGrads[k].Data[j] += g.Data[j]
			}
			k++

			_, g = layer.GetBiasesWithGradient()
			for j := 0; j < len(g.Data); j++ {
				t.SumGrads[k].Data[j] += g.Data[j]
			}
			k++

			layer.ResetGradients()
		}
	}

	return t.Output
}

func (t *VanillaSGDExtBatch) UpdateWeights() {
	if t.BatchIndex == t.BatchSize {
		t.BatchIndex = 0
	} else {
		return
	}

	batchRate := 1 / float64(t.BatchSize)
	k := 0
	for i := 0; i < t.Network.GetLayersCount(); i++ {
		layer, ok := t.Network.GetLayer(i).(nnet.TrainableLayer)
		if ok {
			if !layer.Mutable() {
				k += 2
				continue
			}

			w, _ := layer.GetWeightsWithGradient()
			for j := 0; j < len(w.Data); j++ {
				//value := t.Gradients[k].Data[j]*t.Momentum*batchRate + t.SumGrads[k].Data[j]*batchRate*t.Learning
				value := t.Gradients[k].Data[j]*t.Momentum + t.SumGrads[k].Data[j]*batchRate*t.Learning

				w.Data[j] -= value
				t.Gradients[k].Data[j] = value
				t.SumGrads[k].Data[j] = 0
			}
			k++

			w, _ = layer.GetBiasesWithGradient()
			for j := 0; j < len(w.Data); j++ {
				//value := t.Gradients[k].Data[j]*t.Momentum*batchRate + t.SumGrads[k].Data[j]*batchRate*t.Learning
				value := t.Gradients[k].Data[j]*t.Momentum + t.SumGrads[k].Data[j]*batchRate*t.Learning

				w.Data[j] -= value
				t.Gradients[k].Data[j] = value
				t.SumGrads[k].Data[j] = 0
			}
			k++
		}
	}
}
