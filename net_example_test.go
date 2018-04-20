package nnet_test

import (
	"github.com/drdreyworld/nnet"
	"github.com/drdreyworld/nnet/activation"
	"github.com/drdreyworld/nnet/layer"
	"github.com/drdreyworld/nnet/loss"
	"github.com/drdreyworld/nnet/storage"
	"github.com/drdreyworld/nnet/trainer"
	"log"
	"testing"
)

// Feed forawrd net for solve XOR function
func ExampleNet() {
	xor := &nnet.NetDefault{}

	// create storage for load/save network config
	Storage := storage.JsonFile{Filename: "/tmp/xor.json"}
	// setup network for load/save config
	Storage.SetNet(xor)

	// try to load network from file /tmp/xor.json
	if err := Storage.Load(); err != nil {
		log.Println("load network error:", err.Error())
		log.Println("initialize new network")

		xor = &nnet.NetDefault{
			// setup input data sizes 2x1x1
			IWidth:  2,
			IHeight: 1,
			IDepth:  1,

			// setup loss function type (need for GetLoss function)
			Loss: loss.LOSS_REGRESSION,

			Layers: nnet.Layers{
				// fully connected layer with output sizes 5x5x5 (125 hidden neurons)
				&layer.Dense{OWidth: 5, OHeight: 5, ODepth: 5},

				// activation layer (no weights - only activation function applied)
				&layer.Activation{
					ActFunc: activation.ACTIVATION_SIGMOID,
				},
				// fully connected layer with output sizes 1x1x1 (1 output neuron)
				&layer.Dense{OWidth: 1, OHeight: 1, ODepth: 1},
				// activation layer (no weights - only activation function applied)
				&layer.Activation{
					ActFunc: activation.ACTIVATION_SIGMOID,
				},
			},
		}
		xor.Init()

		Storage.SetNet(xor)
	}

	// create trainer for learning network with LearningRate 0.1
	Trainer := nnet.Trainer(&trainer.VanilaSGD{Learning: 0.1})
	// setup network for training
	Trainer.SetNet(xor)

	// prepare all possible input data vectors
	inputs := []nnet.Data{
		{Dims: []int{2}, Data: []float64{0, 0}},
		{Dims: []int{2}, Data: []float64{1, 0}},
		{Dims: []int{2}, Data: []float64{0, 1}},
		{Dims: []int{2}, Data: []float64{1, 1}},
	}

	// prepare target outputs, relevant input vectors
	targets := []nnet.Data{
		{Dims: []int{1}, Data: []float64{0}},
		{Dims: []int{1}, Data: []float64{1}},
		{Dims: []int{1}, Data: []float64{1}},
		{Dims: []int{1}, Data: []float64{0}},
	}

	// train network 100000 times
	for epoch := 0; epoch < 10000; epoch++ {
		l := 0.0
		for i := 0; i < len(inputs); i++ {
			outputs := Trainer.Activate(&inputs[i], &targets[i])
			lossval := xor.GetLoss(&targets[i], outputs)

			Trainer.UpdateWeights()
			l += lossval
		}

		l /= 4

		if epoch%1000 == 0 {
			log.Println("loss (avg):", l)
		}
	}

	// check network results on same inputs
	for i := 0; i < len(inputs); i++ {
		outputs := xor.Activate(&inputs[i])

		log.Println(inputs[i].Data[0], "xor", inputs[i].Data[1], "=", targets[i].Data[0], "output:", outputs.Data[0])
	}

	if err := Storage.Save(); err != nil {
		log.Fatal("save network error:", err.Error())
	}
}

func Test_ExampleNet(t *testing.T) {
	ExampleNet()
}
