package nnet

type NetStorage interface {
	SetNet(nn NNet)
	Save() error
	Load() error
}
