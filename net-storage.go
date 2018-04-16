package nnet

type NetStorage interface {
	Save(net NNet) error
	Load(net NNet) error
}
