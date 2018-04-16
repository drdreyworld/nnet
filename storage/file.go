package storage

import (
	"encoding/json"
	"errors"
	"github.com/drdreyworld/nnet"
	"io/ioutil"
	"os"
)

const ERR_FILE_FILENAME_NOT_SET = "storage filename not set"

type JsonFile struct {
	Filename string
}

func (s *JsonFile) Save(n nnet.NNet) (err error) {
	if len(s.Filename) == 0 {
		return errors.New(ERR_FILE_FILENAME_NOT_SET)
	}

	d, err := json.Marshal(n.Serialize())
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

func (s *JsonFile) Load(n nnet.NNet) (err error) {
	if len(s.Filename) == 0 {
		return errors.New(ERR_FILE_FILENAME_NOT_SET)
	}

	d, err := ioutil.ReadFile(s.Filename)
	if err != nil {
		return
	}

	c := nnet.NetConfig{}

	if err = json.Unmarshal(d, &c); err == nil {
		err = n.Init(c)
	}

	return
}
