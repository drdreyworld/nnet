package softmax

import (
	"github.com/drdreyworld/nnet/data"
	"github.com/stretchr/testify/assert"
	"math"
	"testing"
)

func TestLayer_Activate(t *testing.T) {
	type testCase struct {
		inputs *data.Data
		output *data.Data
	}

	dims := []int{3, 1, 1}

	testCases := map[string]testCase{
		"a": {
			inputs: &data.Data{
				Dims: dims,
				Data: []float64{-1, 7, 3},
			},
			output: &data.Data{
				Dims: dims,
				Data: []float64{0.0003293204389638929, 0.9816903928255046, 0.017980286735531543},
			},
		},
		"b": {
			inputs: &data.Data{
				Dims: dims,
				Data: []float64{-30, 255, 17},
			},
			output: &data.Data{
				Dims: dims,
				Data: []float64{1.6829555964029658e-124, 1, 4.344234967880666e-104},
			},
		},
		"c": {
			inputs: &data.Data{
				Dims: dims,
				Data: []float64{0, 0, 0.1},
			},
			output: &data.Data{
				Dims: dims,
				Data: []float64{0.32204346439638987, 0.32204346439638987, 0.3559130712072203},
			},
		},
		"d": {
			inputs: &data.Data{
				Dims: dims,
				Data: []float64{1000, 2000, 3000},
			},
			output: &data.Data{
				Dims: dims,
				Data: []float64{0, 0, 1},
			},
		},
		"e": {
			inputs: &data.Data{
				Dims: dims,
				Data: []float64{1, 2, 3},
			},
			output: &data.Data{
				Dims: dims,
				Data: []float64{0.09003057317038046, 0.24472847105479764, 0.6652409557748218},
			},
		},
	}

	for name, tc := range testCases {
		tc := tc
		t.Run(name, func(t *testing.T) {
			layer := New(OutputSizes(3, 1, 1))
			layer.InitDataSizes(3, 1, 1)

			output := layer.Activate(tc.inputs)
			assert.Equal(t, tc.output, output)

			sum := 0.0
			for i := 0; i < len(output.Data); i++ {
				sum += output.Data[i]
			}

			assert.Equal(t, 1.0, math.Round(sum))
		})
	}
}
