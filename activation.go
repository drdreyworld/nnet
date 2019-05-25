package nnet

type ActivationConstructor func(params interface{}) Activation

var ActivationsRegistry = activationsRegistry{}

type activationsRegistry map[string]ActivationConstructor

type Activation interface {
	SetParams(interface{})
	Forward(v float64) float64
	Backward(v float64) float64
}
