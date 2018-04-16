package nnet

import (
	"math/rand"
	"encoding/gob"
)

func init() {
	gob.Register(Data{})
}

type Data struct {
	Dims []int
	Data []float64
}

func (m *Data) InitVector(w int) {
	m.Dims = []int{w}
	m.Data = make([]float64, w)
}

func (m *Data) Fill(v float64) {
	for i := 0; i < len(m.Data); i++ {
		m.Data[i] = v
	}
}

func (m *Data) InitVectorRandom(w int, min, max float64) {
	m.InitVector(w)
	m.FillRandom(min, max)
}

func (m *Data) InitMatrix(w, h int) {
	m.Dims = []int{w, h}
	m.Data = make([]float64, w*h)
}

func (m *Data) InitMatrixRandom(w, h int, min, max float64) {
	m.InitMatrix(w, h)
	m.FillRandom(min, max)
}

func (m *Data) InitCube(w, h, d int) {
	m.Dims = []int{w, h, d}
	m.Data = make([]float64, w*h*d)
}

func (m *Data) InitTensorRandom(w, h, d int, min, max float64) {
	m.InitCube(w, h, d)
	m.FillRandom(min, max)
}

func (m *Data) InitHiperCube(w, h, d, t int) {
	m.Dims = []int{w, h, d, t}
	m.Data = make([]float64, w*h*d*t)
}

func (m *Data) InitHiperCubeRandom(w, h, d, t int, min, max float64) {
	m.InitHiperCube(w, h, d, t)
	m.FillRandom(min, max)
}

func (m *Data) FillRandom(min, max float64) {
	for i := 0; i < len(m.Data); i++ {
		m.Data[i] = min + (max-min)*rand.Float64()
	}
}

func (m Data) CopyZero() (r *Data) {
	r = &Data{}
	r.Dims = make([]int, len(m.Dims))
	r.Data = make([]float64, len(m.Data))
	copy(r.Dims, m.Dims) // copy struct

	return
}

func (m Data) Copy() (r *Data) {
	r = m.CopyZero()
	copy(r.Data, m.Data)
	return
}

func (m Data) RotateTensorMatrixes() (r *Data) {
	r = m.CopyZero()

	w, h, d := m.Dims[0], m.Dims[1], m.Dims[2]

	for z := 0; z < d; z++ {
		zs := z*w * h
		for yc, yn := 0, h-1; yc < h; yc, yn = yc+1, yn-1 {
			for xc, xn := 0, w-1; xc < w; xc, xn = xc+1, xn-1 {
				r.Data[zs + yn*w + xn] = m.Data[zs + yc*w + xc]
			}
		}
	}
	return
}
