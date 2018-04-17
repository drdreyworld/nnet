package nnet

var ActivationsRegistry = activationsRegistry{}

type activationsRegistry map[string]Activation

type Activation interface {
	Forward(v float64) float64
	Backward(v float64) float64
	Serialize() string
}
