package storage

import (
	"encoding/json"
	"errors"
	"github.com/drdreyworld/nnet"
	"io/ioutil"
	"os"
)

const ERR_FILE_FILENAME_NOT_SET = "storage filename not set"
const ERR_FILE_NETWORK_NOT_SET = "storage network not set"

type JsonFile struct {
	Network nnet.NNet
	Filename string
}

func (s *JsonFile) SetNet(n nnet.NNet) {
	s.Network = n
}

func (s *JsonFile) Save() (err error) {
	if s.Network == nil {
		return errors.New(ERR_FILE_NETWORK_NOT_SET)
	}

	if len(s.Filename) == 0 {
		return errors.New(ERR_FILE_FILENAME_NOT_SET)
	}

	d, err := json.Marshal(s.Network.Serialize())
	if err != nil {
		return
	}

	f, err := os.Create(s.Filename)
	if err != nil {
		return
	}
	defer f.Close()

	_, err = f.Write(d)

	return
}

func (s *JsonFile) Load() (err error) {
	if s.Network == nil {
		return errors.New(ERR_FILE_NETWORK_NOT_SET)
	}

	if len(s.Filename) == 0 {
		return errors.New(ERR_FILE_FILENAME_NOT_SET)
	}

	d, err := ioutil.ReadFile(s.Filename)
	if err != nil {
		return
	}

	c := nnet.NetConfig{}

	if err = json.Unmarshal(d, &c); err == nil {
		err = s.Network.Init(c)
	}

	return
}
