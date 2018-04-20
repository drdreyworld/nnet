package nnet

import (
	"encoding/json"
)

type Layers []Layer

type structToMarshalLayer struct {
	Type string
	Data interface{}
}

func (c Layers) MarshalJSON() ([]byte, error) {
	res := []structToMarshalLayer{}
	for i := 0; i < len(c); i++ {
		res = append(res, structToMarshalLayer{
			Type: (c)[i].GetType(),
			Data: (c)[i],
		})
	}
	return json.Marshal(res)
}

type structToUnmarshalLayer struct {
	Type string
	Data *json.RawMessage
}

func (c Layers) UnmarshalJSON(b []byte) (err error) {
	cfg := []structToUnmarshalLayer{}

	if err = json.Unmarshal(b, &cfg); err != nil {
		return err
	}

	for i := 0; i < len(cfg); i++ {
		l, err := LayersRegistry.Create(cfg[i].Type)
		if err != nil {
			return err
		}

		if cfg[i].Data != nil {
			json.Unmarshal(*cfg[i].Data, l)
		}

		c = append(c, l)
	}

	return
}


