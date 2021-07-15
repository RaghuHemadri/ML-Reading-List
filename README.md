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


## Unsupervised Learning
###   Generative Adversarial Networks

-   [GAN](https://arxiv.org/abs/1406.2661)

-   [DCGAN](https://arxiv.org/abs/1511.06434)

-   [InfoGAN](https://arxiv.org/abs/1606.03657)

-   [Improved GAN](https://arxiv.org/abs/1606.03498)

-   [WGAN](https://arxiv.org/abs/1701.07875)

-   [WGAN-GP](https://arxiv.org/abs/1704.00028)

-   [MMD GAN](https://arxiv.org/abs/1705.08584)

-   [StyleGAN](https://arxiv.org/abs/1812.04948)

-   [PG-GAN](https://arxiv.org/abs/1710.10196)

-   [SA-GAN](https://arxiv.org/abs/1805.08318)

-   [SN-GAN](https://openreview.net/pdf?id=B1QRgziT-)

-   [BigGAN](https://arxiv.org/abs/1809.11096)

-   [S3GAN](https://arxiv.org/abs/1903.02271)

###   Style Transfer and Colorization

-   [pix2pix](https://arxiv.org/abs/1611.07004)

-   [CycleGAN](https://arxiv.org/abs/1703.10593)

-   [Attention-Guided CycleGAN](https://arxiv.org/abs/1806.02311)

-   [vid2vid](https://arxiv.org/abs/1808.06601)

-   [Everybody Dance Now](https://arxiv.org/abs/1808.07371)

-   [Style Transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

-   [Fast Style Transfer](https://arxiv.org/abs/1603.08155)

-   [Image Colorization](https://arxiv.org/abs/1603.08511)

-   [Tracking Emerges](https://arxiv.org/abs/1806.09594)

###   Autoregressive Models

-   [MADE](https://arxiv.org/abs/1502.03509)

-   [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759)

-   [PixelCNN](https://arxiv.org/abs/1606.05328)

-   [PixelCNN++](https://arxiv.org/abs/1701.05517)

-   [PixelSNAIL](https://arxiv.org/abs/1712.09763)

-   [WaveNet](https://arxiv.org/abs/1609.03499)

-   [WaveRNN](https://arxiv.org/abs/1802.08435)

-   [SPN](https://arxiv.org/abs/1812.01608)

-   [Image Transformer](https://arxiv.org/abs/1802.05751)

###   Normalizing Flows

-   [NICE](https://arxiv.org/abs/1410.8516)

-   [RealNVP](https://arxiv.org/abs/1605.08803)

-   [Glow](https://arxiv.org/abs/1807.03039)

-   [WaveGlow](https://arxiv.org/abs/1811.00002)

-   [Flow++](https://arxiv.org/abs/1902.00275)

-   [Parallel WaveNet](https://arxiv.org/abs/1711.10433)

###   Variational AutoEncoders

-   [VAE Tutorial](https://arxiv.org/abs/1606.05908)

-   [beta-VAE](https://openreview.net/forum?id=Sy2fzU9gl)

-   [VQ-VAE](https://arxiv.org/abs/1711.00937)

-   [VQ-VAE-2](https://arxiv.org/abs/1906.00446)

-   [Variational Inference with Flows](https://arxiv.org/abs/1505.05770)

-   [IAF](https://arxiv.org/abs/1606.04934)

-   [Variational Lossy Autoencoder](https://arxiv.org/abs/1611.02731)

###   Other Representation Learning

-   [SDAE](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)

-   [Revisiting Self-Sup Learning](https://arxiv.org/abs/1901.09005)

-   [Puzzle](https://arxiv.org/abs/1505.05192)

-   [RotNet](https://arxiv.org/abs/1803.07728)

-   [CPC](https://arxiv.org/abs/1807.03748)

-   [Imagenet Transfer](https://arxiv.org/abs/1805.08974)

-   [Instagram Models](https://arxiv.org/abs/1805.00932)

###   Evaluating Generative Models

-   [A Note](https://arxiv.org/abs/1511.01844)

-   [A Note on IS](https://arxiv.org/abs/1801.01973)

-   [FID Explanation](https://medium.com/@jonathan_hui/gan-how-to-measure-gan-performance-64b988c47732)

-   [NND](https://openreview.net/forum?id=HkxKH2AcFm)

-   [Pros & Cons of GAN Metrics](https://arxiv.org/abs/1802.03446)

-   [Skill Rating](https://arxiv.org/abs/1808.04888)
