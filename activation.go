package nnet

var Activations = ActivationRegistry{}

type ActivationRegistry map[string]Activation

type Activation interface {
	Forward(v float64) float64
	Backward(v float64) float64
	Serialize() string
}
