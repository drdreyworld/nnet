package selu

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"math"
	"testing"
)

func TestActivation_Forward(t *testing.T) {
	type testCaseData struct {
		value    float64
		expected float64
	}

	type testCase struct {
		koeff float64
		cases map[string]testCaseData
	}

	testCases := []testCase{
		{
			koeff: 1.0,
			cases: map[string]testCaseData{
				"greaterThanZero": {value: 1, expected: 1},
				"equalsZero":      {value: 0.0, expected: 0},
				"lessThanZero":    {value: -0.5, expected: -0.3934693402873666},
				"maxValue":        {value: math.MaxFloat64, expected: math.MaxFloat64},
				"minValue":        {value: -math.MaxFloat64, expected: -1.0},
			},
		},
		{
			koeff: 0.01,
			cases: map[string]testCaseData{
				"greaterThanZero": {value: 1, expected: 1},
				"equalsZero":      {value: 0.0, expected: 0.0099},
				"lessThanZero":    {value: -0.5, expected: 0.005965306597126335},
				"maxValue":        {value: math.MaxFloat64, expected: math.MaxFloat64},
				"minValue":        {value: -math.MaxFloat64, expected: -0.0001},
			},
		},
	}

	for _, test := range testCases {
		for name, tc := range test.cases {
			test, tc := test, tc
			t.Run(fmt.Sprintf("%sWithKoeff%f", name, test.koeff), func(t *testing.T) {
				assert.Equal(t, tc.expected, New(test.koeff).Forward(tc.value))
			})
		}
	}
}

func TestActivation_Backward(t *testing.T) {
	type testCaseData struct {
		value    float64
		expected float64
	}

	type testCase struct {
		koeff float64
		cases map[string]testCaseData
	}

	testCases := []testCase{
		{
			koeff: 1.0,
			cases: map[string]testCaseData{
				"greaterThanZero": {value: 15, expected: 1.0},
				"equalsZero":      {value: 0.0, expected: 1.0},
				"lessThanZero":    {value: -0.5, expected: 0.6065306597126334},
				"maxValue":        {value: math.MaxFloat64, expected: 1},
				"minValue":        {value: -math.MaxFloat64, expected: 0.0},
			},
		},
		{
			koeff: 0.01,
			cases: map[string]testCaseData{
				"greaterThanZero": {value: 15, expected: 1.0},
				"equalsZero":      {value: 0.0, expected: 0.01},
				"lessThanZero":    {value: -0.5, expected: 0.006065306597126334},
				"maxValue":        {value: math.MaxFloat64, expected: 1},
				"minValue":        {value: -math.MaxFloat64, expected: 0.0},
			},
		},
	}

	for _, test := range testCases {
		for name, tc := range test.cases {
			test, tc := test, tc
			t.Run(fmt.Sprintf("%sWithKoeff%f", name, test.koeff), func(t *testing.T) {
				assert.Equal(t, tc.expected, New(test.koeff).Backward(tc.value))
			})
		}
	}
}
