# conv-net-viz

Assignment to re-implement
```
Zeiler, M.D. and Fergus, R., 2014, September. Visualizing and understanding convolutional networks. In European conference on computer vision (pp. 818-833). Springer, Cham. Cited by 2457
```

# Scripts

## simple.py
Visualization for a very small network from: https://github.com/aymericdamien/TensorFlow-Examples/
Python3.5, CUDA 8, Tensorflow 1.0
```
$ python simple.py
$ firefox simple.html // for layer-1 analysis
$ firefox simple2.html // for layer-2 analysis
```
Screenshots:

Layer-1 convolution output
![simple screenshot](simple_screen.png)

Layer-2 convolution output
![simple2 screenshot](simple2_screen.png)

Download full html pages in zip:
[ZIP](https://drive.google.com/file/d/0BwTp6MaUSAahLUYwbzgtTFJ3Tkk/view?usp=sharing)

Graph:
![simple graph](simple_graph.png)

## alex.py
Visualization for the Alex network from: http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ (converted from caffe)

First, please, download their weights:
```
$ wget http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
```

To run the visualisation:
```
$ python alex.py
$ firefox alex.html // for layer-1 analysis
$ firefox alex2.html // for layer-2 analysis
$ firefox alex3.html // for layer-3 analysis
```

Screenshots:

Layer-1 convolution output
![Alex screenshot](alex_screen.png)
Layer-2 convolution output
![Alex screenshot](alex2_screen.png)
Layer-3 convolution output (tends to react on dog, or dog hair? or dog eye?)
![Alex screenshot](alex3_screen.png)

## vgg.py

# Preliminary tests
I experienced what conv2d_transpose is doing exactly

Most inspiring resource: <https://github.com/simo23/conv2d_transpose/blob/master/test.py>
```
$ ut_shape.py
$ ut_conv.py
$ ut_one_layer.py
```
Last scripts reconstructs 5th channel (this script is mainly from the cited resource)
![5th channel](DeconvTest5.png)