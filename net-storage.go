package nnet

type NetStorage interface {
	SetNet(nn Net)
	Save() error
	Load() error
}
