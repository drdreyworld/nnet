package nnet

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"sync"
)

type NNet interface {
	Activate(inputs *Mem) (output *Mem)
	Backprop(deltas *Mem) (gradient *Mem)
	GetOutputDeltas(target, output *Mem) (res *Mem)
	GetLayersCount() int
	GetLayer(index int) Layer
}

type Net struct {
	IWidth  int
	IHeight int
	IDepth  int

	OWidth  int
	OHeight int
	ODepth  int

	layers   []Layer
	filename string

	sync.Mutex
	locked bool
}

func (n *Net) Init(cfg interface{}) (err error) {
	c, err := CreateNetConfig(cfg)
	if err != nil {
		panic(err)
		return
	}

	n.filename = c.Filename

	for _, v := range c.Layers {
		var l Layer

		l, err = Layers.Create(v.Type)
		if err != nil {
			panic(err)
			return
		}

		if err = l.Init(v); err != nil {
			panic(err)
			return
		}

		n.layers = append(n.layers, l)
	}

	n.IWidth, n.IHeight, n.IDepth = c.IWidth, c.IHeight, c.IDepth
	n.OWidth, n.OHeight, n.ODepth = n.initDataSizes(n.IWidth, n.IHeight, n.IDepth)

	return
}

func (n *Net) initDataSizes(w, h, d int) (int, int, int) {
	fmt.Println("layers count: ", len(n.layers))
	for i := 0; i < len(n.layers); i++ {
		fmt.Println("### initDataSizes for layer ", i)
		w, h, d = n.layers[i].InitDataSizes(w, h, d)
	}
	return w, h, d
}

func (n *Net) SetFilename(filename string) {
	n.filename = filename
}

func (n *Net) Activate(inputs *Mem) *Mem {
	n.Lock()
	defer n.Unlock()

	for i := 0; i < len(n.layers); i++ {
		inputs = n.layers[i].Activate(inputs)
	}
	return inputs
}

func (n *Net) GetOutputDeltas(target, output *Mem) (res *Mem) {
	res = target.CopyZero()
	for i := 0; i < len(res.Data); i++ {
		res.Data[i] = -(target.Data[i] - output.Data[i])
	}
	return
}

func (n *Net) Backprop(deltas *Mem) (gradient *Mem) {
	n.Lock()
	defer n.Unlock()

	gradient = deltas.Copy()

	for i := len(n.layers) - 1; i >= 0; i-- {
		gradient = n.layers[i].Backprop(gradient)
	}

	return gradient
}

func (n *Net) GetLossClassification(target, result *Mem) (res float64) {
	for i := 0; i < len(target.Data); i++ {
		if target.Data[i] == 1 {
			return -math.Log(result.Data[i])
		}
	}
	return 0
}

func (n *Net) GetLossRegression(target, result *Mem) (res float64) {
	for i := 0; i < len(target.Data); i++ {
		res += math.Pow(result.Data[i]-target.Data[i], 2)
	}
	return 0.5 * res
}

func (n *Net) Serialize() NetConfig {
	res := NetConfig{}

	res.IWidth, res.IHeight, res.IDepth = n.IWidth, n.IHeight, n.IDepth
	res.OWidth, res.OHeight, res.ODepth = n.OWidth, n.OHeight, n.ODepth

	for i := 0; i < len(n.layers); i++ {
		res.Layers = append(res.Layers, n.layers[i].Serialize())
	}
	return res
}

func (n *Net) setLocked() (res bool) {
	n.Lock()

	if res = !n.locked; res {
		n.locked = true
	}

	n.Unlock()
	return
}

func (n *Net) setUnLocked() (res bool) {
	n.Lock()

	if res = n.locked; res {
		n.locked = false
	}

	n.Unlock()
	return
}

func (n *Net) SaveConfig() (err error) {
	n.Lock()
	defer n.Unlock()

	if len(n.filename) == 0 {
		return errors.New("Net filename is not set")
	}

	d, err := json.Marshal(n.Serialize())
	if err != nil {
		return
	}

	f, err := os.Create(n.filename)
	if err != nil {
		return
	}
	defer f.Close()

	_, err = f.Write(d)

	return
}

func (n *Net) LoadConfig() (err error) {
	n.Lock()
	defer n.Unlock()

	if len(n.filename) == 0 {
		return errors.New("Net filename is not set")
	}

	d, err := ioutil.ReadFile(n.filename)
	if err != nil {
		return
	}

	c := NetConfig{}

	if err = json.Unmarshal(d, &c); err == nil {
		err = n.Init(c)
	}

	return
}

func (n *Net) GetLayersCount() int {
	return len(n.layers)
}

func (n *Net) GetLayer(index int) Layer {
	if index > -1 && index < len(n.layers) {
		return n.layers[index]
	}
	return nil
}
