# fcn_caffe_cpp_forward
fcn_caffe_cpp_forward



Already have a lot of program of fcn forward by python.This is the cpp program of fcn to forward use caffe.

My cpp is just two class,use fcn32s deploy and the last feature map is 256*256.

You need modify this program to adapt you situation.

There are the deploy and the cpp.


The Compile command:

g++ -o fcntest main.cpp `pkg-config --cflags --libs opencv`  -I /***path to you caffe***/include  -I/***path to you caffe***/build/src -L /***path to you caffe***/build/lib -I /usr/local/cuda/include  -L /usr/local/cuda/lib64 -lcaffe -lglog  -lboost_system  -lcudnn
