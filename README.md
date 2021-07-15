# ML Reading List
## ImageNet Architectures and Tricks

### Architectures

-   [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

-   [GoogLeNet](https://arxiv.org/abs/1409.4842)

-   [Inception-v2/v3](https://arxiv.org/abs/1512.00567)

-   [ResNet](https://arxiv.org/abs/1512.03385)

-   [ResNet-v2](https://arxiv.org/abs/1603.05027)

-   [ResNeXt](https://arxiv.org/abs/1611.05431)

-   [VGG](https://arxiv.org/abs/1409.1556)

-   [EfficientNet](https://arxiv.org/abs/1905.11946)

### Tricks

-   [Batch Norm](https://arxiv.org/abs/1502.03167)

-   [Dropout](http://jmlr.org/papers/v15/srivastava14a.html)

-   [Layer Norm](https://arxiv.org/abs/1607.06450)

-   [Group Norm](https://arxiv.org/abs/1803.08494)

-   [Squeeze & Excitation](https://arxiv.org/abs/1709.01507)

-   [Xception](https://arxiv.org/abs/1610.02357)

### Other

-   [ImageNet Dataset](http://www.image-net.org/papers/imagenet_cvpr09.pdf)

## Object Detection and Segmentation

###   Shallow Stuff

-   [Selective Search](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)

-   [DPM](http://cs.brown.edu/people/pfelzens/papers/lsvm-pami.pdf)

-   [HOG](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)

###   2D

-   [R-CNN](https://arxiv.org/abs/1311.2524)

-   [Fast R-CNN](https://arxiv.org/abs/1504.08083)

-   [Faster R-CNN](https://arxiv.org/abs/1506.01497)

-   [Mask R-CNN](https://arxiv.org/abs/1703.06870)

-   [Yolo](https://arxiv.org/abs/1506.02640)

-   [SSD](https://arxiv.org/abs/1512.02325)

-   [Fast Yolo](https://arxiv.org/abs/1709.05943)

-   [U-Net](https://arxiv.org/abs/1505.04597)

-   [DeepLab-v3](https://arxiv.org/abs/1706.05587)

###   3D

-   [PointNet](https://arxiv.org/abs/1612.00593)

-   [Mesh R-CNN](https://arxiv.org/abs/1906.02739)

-   [PointNet++](https://arxiv.org/abs/1706.02413)

-   [Frustum PointNet](https://arxiv.org/abs/1711.08488)

-   [VoxelNet](https://arxiv.org/abs/1711.06396)

## Systems for ML

###  Overviews

-   [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)

-   [A Berkeley View of Systems Challenges for AI](https://arxiv.org/abs/1712.05855)

###   Small Models

-   [MobileNet](https://arxiv.org/abs/1704.04861)

-   [MobileNet-v2](https://arxiv.org/abs/1801.04381)

-   [MobileNet-v3](https://arxiv.org/abs/1905.02244)

-   [SqueezeNet](https://arxiv.org/abs/1602.07360)

-   [EfficientNet](https://arxiv.org/abs/1905.11946)

-   [SqueezeDet](https://arxiv.org/abs/1612.01051)

###   Hyperparameter Search

-   [Hyperband](https://arxiv.org/abs/1603.06560)

-   [ASHA](https://arxiv.org/abs/1810.05934)

-   [PBT](https://arxiv.org/abs/1711.09846)

###   Quantization

-   [A Survey on Methods and Theories of Quantized Neural Networks](https://arxiv.org/abs/1808.04752)

-   [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)

-   [Fixed Point Quantization of Deep Convolutional Networks](https://arxiv.org/abs/1511.06393)

-   [Mixed Precision Quantization of ConvNets via Differentiable Neural Architecture Search](https://arxiv.org/abs/1812.00090)

-   [XNOR-Net](https://arxiv.org/abs/1603.05279)

###   Model Serving

-   [InferLine](https://ucbrise.github.io/cs294-ai-sys-fa19/assets/preprint/inferline_draft.pdf)

###   Graph Compilation

-   [TVM](https://arxiv.org/abs/1802.04799)

###   Distributed Training

####   Background

-   [HogWild](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf)

-   [Parameter Server](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf)

-   [AllReduce](http://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)

-   [Data Parallel vs Model Parallel](https://en.wikipedia.org/w/index.php?title=Data_parallelism&oldid=807618997#Data_Parallelism_vs._Model_Parallelism[4])

-   [Distributed Overview](https://arxiv.org/abs/1802.09941)

####   Data Parallelism

-   [Horovod](https://arxiv.org/abs/1802.05799)

####   Model Parallelism

-   [GPipe](https://arxiv.org/abs/1811.06965)

-   [PipeDream](https://cs.stanford.edu/~matei/papers/2019/sosp_pipedream.pdf)

####   Some Tricks

-   [Gradient Compression](https://arxiv.org/abs/1802.07389)

-   [Gradent Sparsification](https://arxiv.org/abs/1710.09854)

####   ImageNet in Minutes

-   [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)

-   [With Hierarchical All-Reduce](https://arxiv.org/abs/1807.11205)

####   Some Tools

-   [Horovod](https://github.com/horovod/horovod)

-   [BytePS](https://github.com/bytedance/byteps)

-   [TensorFlow Distributed](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy)

-   [PyTorch Distributed DataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)

###   Low Memory

####   Training

-   [RevNet](https://arxiv.org/abs/1707.04585)

-   [Gradient Checkpointing](https://arxiv.org/pdf/1904.10631.pdf)
