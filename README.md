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
$ firefox simple.html
```
Screenshot:

![simple screenshot](simple_screen.png)

Graph:
![simple graph](simple_graph.png)

## alex.py
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