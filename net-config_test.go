package nnet

import "testing"

func TestNetConfig_CreateLayers(t *testing.T) {
	LayersRegistry["testNetConfigLayer1"] = testNetConfigLayer1Constructor

	cfg := NetConfig{
		IWidth:  1,
		IHeight: 2,
		IDepth:  3,

		OWidth:  4,
		OHeight: 8,
		ODepth:  12,

		Layers:  []LayerConfig{
			{
				Type: "testNetConfigLayer1",
				Data: LayerConfigData{
					"ParamA" : "Param for first layer",
				},
			},
			{
				Type: "testNetConfigLayer1",
				Data: LayerConfigData{
					"ParamA" : "Param for second layer",
				},
			},
		},
	}

	layers, err := cfg.CreateLayers()
	if err != nil {
		t.Error("create layers error:", err.Error())
	}

	if layers == nil || len(layers) != 2 {
		t.Error("invalid created layers slice")
	}

	if lr, ok := layers[0].(*testNetConfigLayer1); !ok {
		t.Error("invalid layer type")
	} else {
		if lr.paramA != "Param for first layer" {
			t.Error("missed layer param ParamA")
		}

		if lr.iw != 1 || lr.ih != 2 || lr.id != 3 {
			t.Error("invalid layer input sizes")
		}

		if lr.ow != 2 || lr.oh != 4 || lr.od != 6 {
			t.Error("invalid layer output sizes")
		}

		if lr.output == nil || lr.output.Data == nil {
			t.Error("invalid layer output")
		}

		if len(lr.output.Data) != 2*4*6 {
			t.Error("invalid layer output size")
		}
	}

	if lr, ok := layers[1].(*testNetConfigLayer1); !ok {
		t.Error("invalid layer type")
	} else {
		if lr.paramA != "Param for second layer" {
			t.Error("missed layer param ParamA")
		}

		if lr.iw != 2 || lr.ih != 4 || lr.id != 6 {
			t.Error("invalid layer input sizes")
		}

		if lr.ow != 4 || lr.oh != 8 || lr.od != 12 {
			t.Error("invalid layer output sizes")
		}

		if lr.output == nil || lr.output.Data == nil {
			t.Error("invalid layer output")
		}

		if len(lr.output.Data) != 4*8*12 {
			t.Error("invalid layer output size")
		}
	}
}


func TestNetConfig_CreateLayersError(t *testing.T) {
	LayersRegistry["testNetConfigLayer1"] = testNetConfigLayer1Constructor

	cfg := NetConfig{
		IWidth:  1,
		IHeight: 2,
		IDepth:  3,

		OWidth:  4,
		OHeight: 8,
		ODepth:  12,

		Layers:  []LayerConfig{
			{
				Type: "testNetConfigLayer1",
				Data: LayerConfigData{
					"ParamA" : "Param for first layer",
				},
			},
			{
				Type: "invalidKey2",
				Data: LayerConfigData{
					"ParamA" : "Param for second layer",
				},
			},
		},
	}

	layers, err := cfg.CreateLayers()
	if err == nil {
		t.Error("layer created with invalid types")
	}

	if layers != nil {
		t.Error("layers partially filled")
	}
}

func testNetConfigLayer1Constructor(cfg LayerConfig) (res Layer, err error) {
	res = &testNetConfigLayer1{}
	err = res.Unserialize(cfg)
	return
}

type testNetConfigLayer1 struct {
	iw, ih, id int
	ow, oh, od int

	inputs *Data
	output *Data
	paramA string
}

func (t *testNetConfigLayer1) InitDataSizes(w, h, d int) (int, int, int) {
	t.iw, t.ih, t.id = w, h, d
	t.ow, t.oh, t.od = 2*w, 2*h, 2*d

	t.output = &Data{}
	t.output.InitCube(t.ow, t.oh, t.od)

	return t.ow, t.oh, t.od
}

func (t *testNetConfigLayer1) Activate(inputs *Data) (output *Data) {
	t.inputs = inputs
	return t.output
}

func (t *testNetConfigLayer1) Backprop(deltas *Data) (nextDeltas *Data) {
	return deltas.Copy()
}

func (t *testNetConfigLayer1) Unserialize(cfg LayerConfig) (err error) {
	if err = cfg.CheckType("testNetConfigLayer1"); err == nil {
		t.paramA = cfg.Data.String("ParamA")
	}
	return
}

func (t *testNetConfigLayer1) Serialize() (cfg LayerConfig) {
	cfg.Type = "testNetConfigLayer1"
	cfg.Data = LayerConfigData{
		"ParamA": t.paramA,
	}
	return
}

func (t *testNetConfigLayer1) GetOutput() *Data {
	return t.output
}
