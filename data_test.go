package nnet

import (
	"math/rand"
	"testing"
)

func testDataOnCorrectDimentionsCount(t *testing.T, d *Data, i int) {
	if len(d.Dims) != i {
		t.Error("incorrect Dims length")
	}
}

func testDataOnCorrectWidth(t *testing.T, d *Data, i int) {
	if d.Dims[0] != i {
		t.Error("incorrect Dims.Width")
	}
}

func testDataOnCorrectHeight(t *testing.T, d *Data, i int) {
	if d.Dims[1] != i {
		t.Error("incorrect Dims.Height")
	}
}

func testDataOnCorrectDepth(t *testing.T, d *Data, i int) {
	if d.Dims[2] != i {
		t.Error("incorrect Dims.Depth")
	}
}

func testDataOnCorrectCubesCount(t *testing.T, d *Data, i int) {
	if d.Dims[3] != i {
		t.Error("incorrect Dims.CubesCount")
	}
}

func testDataOnCorrectDataSize(t *testing.T, d *Data, size int) {
	t.Helper()
	if len(d.Data) != size {
		t.Error("incorrect Data length")
	}
}

func testDataIsFilledByValue(t *testing.T, d *Data, v float64) {
	t.Helper()

	for i := 0; i < len(d.Data); i++ {
		if d.Data[i] != v {
			t.Fatal("incorrect value in data")
		}
	}
}

func testDataIsRandomInitialized(t *testing.T, d, b *Data) {
	t.Helper()

	equal := true

	for i := 0; i < len(d.Data) && equal; i++ {
		equal = d.Data[i] == b.Data[i]
	}

	if equal {
		t.Error("initialized equal random data")
	}
}

func testDataOnCorrectVector(t *testing.T, d *Data, size int) {
	t.Helper()
	testDataOnCorrectDimentionsCount(t, d, 1)
	testDataOnCorrectWidth(t, d, size)
	testDataOnCorrectDataSize(t, d, size)
}

func testDataOnCorrectMatrix(t *testing.T, d *Data, w, h int) {
	t.Helper()
	testDataOnCorrectDimentionsCount(t, d, 2)
	testDataOnCorrectWidth(t, d, w)
	testDataOnCorrectHeight(t, d, h)
	testDataOnCorrectDataSize(t, d, w*h)
}

func testDataOnCorrectCube(t *testing.T, c *Data, w, h, d int) {
	t.Helper()
	testDataOnCorrectDimentionsCount(t, c, 3)
	testDataOnCorrectWidth(t, c, w)
	testDataOnCorrectHeight(t, c, h)
	testDataOnCorrectDepth(t, c, d)
	testDataOnCorrectDataSize(t, c, w*h*d)
}

func testDataOnCorrectHiperCube(t *testing.T, c *Data, w, h, d, k int) {
	t.Helper()
	testDataOnCorrectDimentionsCount(t, c, 4)
	testDataOnCorrectWidth(t, c, w)
	testDataOnCorrectHeight(t, c, h)
	testDataOnCorrectDepth(t, c, d)
	testDataOnCorrectCubesCount(t, c, k)
	testDataOnCorrectDataSize(t, c, w*h*d*k)
}

func testMinMaxValues(t *testing.T, min, max, amin, amax float64) {
	t.Helper()

	if min != amin {
		t.Error("min value is incorrect")
	}

	if max != amax {
		t.Error("min value is incorrect")
	}
}

func TestData_Fill(t *testing.T) {
	v := rand.Float64()

	d := &Data{}
	d.InitVector(15)
	d.Fill(v)

	testDataOnCorrectVector(t, d, 15)
	testDataIsFilledByValue(t, d, v)

	v = 0.15
	d.InitVector(21)
	d.Fill(v)

	testDataOnCorrectVector(t, d, 21)
	testDataIsFilledByValue(t, d, v)
}

func TestData_FillRandom(t *testing.T) {
	d := &Data{}
	d.InitVectorRandom(10, -0.1, 0.001)
	r := false
	for i := 0; i < len(d.Data); i++ {
		if d.Data[i] < -0.1 {
			t.Error("less than min value")
		}
		if d.Data[i] > -0.001 {
			t.Error("greater than max value")
		}
		r = d.Data[i] != 0 || r
	}

	if !r {
		t.Error("all values is 0")
	}
}

func TestData_InitVector(t *testing.T) {
	d := &Data{}
	d.InitVector(7)
	testDataOnCorrectVector(t, d, 7)

	d.InitVector(15)
	testDataOnCorrectVector(t, d, 15)

	d.InitVector(0)
	testDataOnCorrectVector(t, d, 0)
}

func TestData_InitVectorRandom(t *testing.T) {
	d := &Data{}
	d.InitVectorRandom(7, -10, 17)
	testDataOnCorrectVector(t, d, 7)

	b := &Data{}
	b.InitVectorRandom(7, -10, 17)
	testDataOnCorrectVector(t, b, 7)

	testDataIsRandomInitialized(t, d, b)
}

func TestData_InitMatrix(t *testing.T) {
	d := &Data{}
	d.InitMatrix(7, 2)
	testDataOnCorrectMatrix(t, d, 7, 2)
	testDataIsFilledByValue(t, d, 0)

	d.InitMatrix(4, 17)
	d.Fill(0.8)
	testDataOnCorrectMatrix(t, d, 4, 17)
	testDataIsFilledByValue(t, d, 0.8)
}

func TestData_InitMatrixRandom(t *testing.T) {
	d := &Data{}
	d.InitMatrixRandom(7, 3, -1, 3)
	testDataOnCorrectMatrix(t, d, 7, 3)

	b := &Data{}
	b.InitMatrixRandom(7, 3, -1, 3)
	testDataOnCorrectMatrix(t, b, 7, 3)

	testDataIsRandomInitialized(t, d, b)
}

func TestData_InitCube(t *testing.T) {
	d := &Data{}
	d.InitCube(3, 4, 2)
	testDataIsFilledByValue(t, d, 0)
	testDataOnCorrectCube(t, d, 3, 4, 2)

	d.InitCube(18, 6, 1)
	d.Fill(21)
	testDataIsFilledByValue(t, d, 21)
	testDataOnCorrectCube(t, d, 18, 6, 1)
}

func TestData_InitCubeRandom(t *testing.T) {
	d := &Data{}
	d.InitCubeRandom(3, 4, 5, 0.1, 0.2)
	testDataOnCorrectCube(t, d, 3, 4, 5)

	b := &Data{}
	b.InitCubeRandom(3, 4, 5, 0.1, 0.2)
	testDataOnCorrectCube(t, d, 3, 4, 5)
	testDataIsRandomInitialized(t, d, b)
}

func TestData_InitHiperCube(t *testing.T) {
	d := &Data{}
	d.InitHiperCube(3, 4, 2, 17)
	testDataIsFilledByValue(t, d, 0)
	testDataOnCorrectHiperCube(t, d, 3, 4, 2, 17)

	d.InitHiperCube(18, 6, 1, 7)
	d.Fill(21)
	testDataIsFilledByValue(t, d, 21)
	testDataOnCorrectHiperCube(t, d, 18, 6, 1, 7)
}

func TestData_InitHiperCubeRandom(t *testing.T) {
	d := &Data{}
	d.InitHiperCubeRandom(3, 4, 5, 7, 0.1, 0.2)
	testDataOnCorrectHiperCube(t, d, 3, 4, 5, 7)

	b := &Data{}
	b.InitHiperCubeRandom(3, 4, 5, 7, 0.1, 0.2)
	testDataOnCorrectHiperCube(t, d, 3, 4, 5, 7)
	testDataIsRandomInitialized(t, d, b)
}

func TestData_CopyZero(t *testing.T) {
	d := &Data{}
	d.InitMatrixRandom(7, 5, -0.5, 0.7)

	b := d.CopyZero()
	testDataOnCorrectMatrix(t, b, 7, 5)
	testDataIsFilledByValue(t, b, 0)
}

func TestData_Copy(t *testing.T) {
	d := &Data{}
	d.InitMatrixRandom(7, 5, -0.5, 0.7)

	b := d.Copy()
	testDataOnCorrectMatrix(t, b, 7, 5)

	d.Data[3] = 17
	b.Data[3] = 6

	if d.Data[3] == b.Data[3] {
		t.Error("copied data is link on source data")
	}

	for i := 0; i < len(d.Data); i++ {
		if i == 3 {
			continue
		}

		if b.Data[i] != d.Data[i] {
			t.Error("copied data values mistmatch with source data")
		}
	}

	d.Dims[0] = 8
	b.Dims[0] = 3

	if d.Dims[0] == b.Dims[0] {
		t.Error("copied dims is link on source dims")
	}
}

func TestData_RotateTensorMatrixes(t *testing.T) {
	d := &Data{}
	d.InitCube(3, 3, 3)
	d.Data = []float64{
		10, 11, 12,
		13, 14, 15,
		16, 17, 18,

		20, 21, 22,
		23, 24, 25,
		26, 27, 28,

		30, 31, 32,
		33, 34, 35,
		36, 37, 38,
	}

	b := &Data{}
	b.InitCube(3, 3, 3)
	b.Data = []float64{
		18, 17, 16,
		15, 14, 13,
		12, 11, 10,

		28, 27, 26,
		25, 24, 23,
		22, 21, 20,

		38, 37, 36,
		35, 34, 33,
		32, 31, 30,
	}

	d.RotateMatrixesInCube()

	for i := 0; i < len(d.Data); i++ {
		if d.Data[i] != b.Data[i] {
			t.Error("rotated data is incorrect")
		}
	}
}

func TestData_Rotate(t *testing.T) {
	d := &Data{}
	d.InitVector(3)
	d.Data = []float64{1, 2, 3}

	b := &Data{}
	b.InitVector(3)
	b.Data = []float64{3, 2, 1}

	d.Rotate(0, 3)

	for i := 0; i < len(d.Data); i++ {
		if d.Data[i] != b.Data[i] {
			t.Error("rotated data is incorrect")
		}
	}

	d.Data = []float64{1}
	b.Data = []float64{1}

	d.Rotate(0, 1)

	for i := 0; i < len(d.Data); i++ {
		if d.Data[i] != b.Data[i] {
			t.Error("rotated data is incorrect")
		}
	}

	d.Data = []float64{1, 2}
	b.Data = []float64{2, 1}

	d.Rotate(0, 2)

	for i := 0; i < len(d.Data); i++ {
		if d.Data[i] != b.Data[i] {
			t.Error("rotated data is incorrect")
		}
	}

	d.Data = []float64{1, 2, 3, 4}
	b.Data = []float64{4, 3, 2, 1}

	d.Rotate(0, 4)

	for i := 0; i < len(d.Data); i++ {
		if d.Data[i] != b.Data[i] {
			t.Error("rotated data is incorrect")
		}
	}
}

func TestData_GetMinMaxValues(t *testing.T) {
	d := &Data{}
	d.InitVector(4)
	d.Data = []float64{-7, 3, 6, 15}

	min, max := d.GetMinMaxValues(0, 4)
	testMinMaxValues(t, min, max, -7, 15)

	d.Data = []float64{3, -7, 15, 6}

	min, max = d.GetMinMaxValues(0, 4)
	testMinMaxValues(t, min, max, -7, 15)

	d.Data = []float64{15, 3, 6, -7}

	min, max = d.GetMinMaxValues(0, 4)
	testMinMaxValues(t, min, max, -7, 15)
}
