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
## Theory
###   Generalization
-   [Rethinking Generalization](https://arxiv.org/abs/1611.03530)
-   [Large-Batch Training](https://arxiv.org/abs/1609.04836)
-   [Sharp Minima](https://arxiv.org/abs/1703.04933)
-   [Parameter-Function Map](https://arxiv.org/abs/1805.08522)
###   Robustness
####   Adversarial Robustness
-   [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644)
-   [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)
-   [Robustness May Be At Odds With Accuracy](https://arxiv.org/abs/1805.12152)
-   [Adversarially Robust Generalization Requires More Data](https://arxiv.org/abs/1804.11285)
-   [Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/abs/1905.02175)
-   [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/abs/1902.02918)
####   Non-Adversarial Robustness
-   [Generalisation in Humans and Deep Neural Networks](https://arxiv.org/abs/1808.08750)
-   [Benchmarking Neural Network Robustness to Common Corruptions and Surface Variations](https://arxiv.org/abs/1807.01697)
-   [Texture Bias](https://arxiv.org/abs/1811.12231)
-   [Do ImageNet Classifiers Generalize to ImageNet?](https://arxiv.org/abs/1902.10811)
-   [When Robustness Doesn't Promote Robustness](https://openreview.net/forum?id=HyxPIyrFvH)
-   [Should Adversarial Attacks Use Pixel p-Norm?](https://arxiv.org/abs/1906.02439)
-   [Transfer of Adversarial Robustness Between Perturbation Types](https://arxiv.org/abs/1905.01034)
###   Optimization
####   Optimizers
-   [Adagrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
-   [Adadelta](https://arxiv.org/abs/1212.5701)
-   [RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
-   [Adam](https://arxiv.org/abs/1412.6980)
-   [RAdam](https://arxiv.org/abs/1908.03265)
-   [Lookahead](https://arxiv.org/abs/1907.08610)
####   Non-Optimizers
-   [One Neuron](https://arxiv.org/abs/1805.08671)
-   [Goldilocks Zone](https://arxiv.org/abs/1807.02581)
-   [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
-   [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)
-   [Qualitatively Characterizing Neural Network Optimization Problems](https://arxiv.org/abs/1412.6544)
-   [Large-Batch Training](https://arxiv.org/abs/1812.06162)
-   [On The Marginal Value of Adaptive Gradient Methods in Machine Learning](https://arxiv.org/abs/1705.08292)
## Neural Architecture Search
###   Random Search
-   [Random Search](https://arxiv.org/abs/1902.07638)
###   RL Based
-   [NAS with RL](https://arxiv.org/abs/1611.01578)
-   [Optimizer Search](https://arxiv.org/abs/1709.07417)
-   [ENAS](https://arxiv.org/abs/1802.03268)
-   [NASNet](https://arxiv.org/abs/1707.07012)
-   [MobileNet-v3](https://arxiv.org/abs/1905.02244)
###   Hierarchial
-   [Hierarchical Representations](https://arxiv.org/abs/1711.00436)
-   [PNAS](https://arxiv.org/abs/1712.00559)
###   Differentiable
-   [DARTS](https://arxiv.org/abs/1806.09055)
-   [DARTS+](https://arxiv.org/abs/1909.06035)
###   Evolutionary
-   [AmoebaNet](https://arxiv.org/abs/1802.01548)
###   Other Architecture Search
-   [Weight Sharing](http://proceedings.mlr.press/v80/bender18a/bender18a.pdf)
-   [MorphNet](https://arxiv.org/abs/1711.06798)
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
## Reinforcement Learning
###   Q Learning
-   [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
-   [DDPG](https://arxiv.org/abs/1509.02971)
-   [TD3](https://arxiv.org/abs/1802.09477)
-   [SAC](https://arxiv.org/abs/1801.01290)
###   Policy Gradients
-   [REINFORCE](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
-   [TRPO](https://arxiv.org/abs/1502.05477)
-   [PPO](https://arxiv.org/abs/1707.06347)
###   Model Based RL
-   [GPS](https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf)
-   [MBPO](https://arxiv.org/abs/1906.08253)
-   [DSAE](https://arxiv.org/abs/1509.06113)
-   [PETS](https://arxiv.org/abs/1805.12114)
-   [POPLIN](https://arxiv.org/abs/1906.08649)
###   Meta RL
-   [MAML](https://arxiv.org/abs/1703.03400)
-   [PEARL](https://arxiv.org/abs/1903.08254)
###   Off-Policy RL
-   [BEAR](https://arxiv.org/abs/1906.00949)
-   [BCQ](https://arxiv.org/abs/1812.02900)
###   Goal Conditioned RL
-   [HER](https://arxiv.org/abs/1707.01495)
-   [UVF](http://proceedings.mlr.press/v37/schaul15.pdf)
-   [RIG](https://arxiv.org/abs/1807.04742)
-   [Goal GAN](https://arxiv.org/abs/1705.06366)
###   Inverse RL / Reward Learning
-   [GAIL](https://arxiv.org/abs/1606.03476)
-   [VICE](https://arxiv.org/abs/1805.11686)
-   [Maximum Entropy Inverse Reinforcement Learning](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf)
-   [Maximum Entropy Deep Inverse Reinforcement Learning](https://arxiv.org/abs/1507.04888)
###   Imitation Learning
-   [Behavioral Cloning](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_69)
-   [DAgger](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf)
-   [MIL](https://arxiv.org/abs/1709.04905)
###   Exploration
-   [Curiosity](https://pathak22.github.io/noreward-rl/resources/icml17.pdf)
-   [Count-Based Exploration](https://arxiv.org/abs/1606.01868)
###   Self-Play
-   [AlphaGo](https://www.nature.com/articles/nature16961)
-   [AlphaZero](https://arxiv.org/abs/1712.01815)
-   [Asymmetric Self-Play](https://arxiv.org/abs/1703.05407)
-   [Multi-Agent Competition](https://arxiv.org/abs/1710.03748)
-   [OpenAI Hide-and-Seek](https://arxiv.org/abs/1909.07528)
## Sequential / NLP
###   Word Vectors
-   [Word2Vec](https://arxiv.org/abs/1301.3781)
-   [GloVe](https://nlp.stanford.edu/pubs/glove.pdf)
-   [Poincare GloVe](https://arxiv.org/abs/1810.06546)
###   Attention
-   [Align & Translate](https://arxiv.org/abs/1409.0473)
-   [ByteNet](https://arxiv.org/abs/1610.10099)
-   [Transformer](https://arxiv.org/abs/1706.03762)
-   [Transformer-XL](https://arxiv.org/abs/1901.02860)
-   [Reformer](https://openreview.net/forum?id=rkgNKkHtvB)
###   Unsupervised
-   [BERT](https://arxiv.org/abs/1810.04805)
-   [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
-   [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
-   [XLNet](https://arxiv.org/abs/1906.08237)
## Image Captioning / VQA
###   Image Captioning Datasets
-   [COCO](https://arxiv.org/abs/1405.0312)
-   [Flick30k Entities](https://arxiv.org/abs/1505.04870)
-   [DAQUAR](https://papers.nips.cc/paper/5411-a-multi-world-approach-to-question-answering-about-real-world-scenes-based-on-uncertain-input.pdf)
-   [Conceptual Captions](https://www.aclweb.org/anthology/P18-1238.pdf)
###  Visual Question Answering Datasets
-   [Visual7W](https://arxiv.org/abs/1511.03416)
-   [Original VQA Paper](https://arxiv.org/abs/1505.00468)
-   [Visual Madlibs](https://arxiv.org/abs/1506.00278)
-   [COCO-QA](https://arxiv.org/abs/1505.02074)
-   [CLEVR](https://arxiv.org/abs/1612.06890)
###   Evaluation Metrics
-   [BLEU](https://dl.acm.org/citation.cfm?id=1073135)
-   [Meteor](https://dl.acm.org/citation.cfm?id=1626389)
-   [ROUGE](https://www.aclweb.org/anthology/W04-1013/)
-   [CIDEr](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf)
-   [SPICE](https://arxiv.org/abs/1607.08822)
-   [Policy Gradients for SPIDEr](https://arxiv.org/abs/1612.00370)
-   [Word Mover's Distance](https://www.aclweb.org/anthology/E17-1019/)
###   Attention Models
-   [Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning](https://arxiv.org/abs/1612.01887)
-   [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)
-   [Show, Attend, and Tell](https://arxiv.org/abs/1502.03044)
-   [Conceptual Captions (Transformer Language Model)](https://www.aclweb.org/anthology/P18-1238.pdf)
-   [FiLM](https://arxiv.org/abs/1709.07871)
-   [Incorporating Copying Mechanism in Image Captioning for Learning Novel Objects](https://arxiv.org/abs/1708.05271)
###   Modular Architectures
-   [Neural Baby Talk](https://arxiv.org/abs/1803.09845)
-   [Neural Module Networks](https://arxiv.org/abs/1511.02799)
-   [Learning to Compose Neural Networks for Question Answering](https://arxiv.org/abs/1601.01705)
-   [Learning to Reason: End-to-End Module Networks for Visual Question Answering](https://arxiv.org/abs/1704.05526)
-   [Modeling Relationships in Referential Expressions with Compositional Modular Networks](https://arxiv.org/abs/1611.09978)
###   Foundations
-   [Baby Talk](http://tamaraberg.com/papers/generation_cvpr11.pdf)
-   [Show and Tell](https://arxiv.org/abs/1411.4555)
-   [Long-term Recurrent Convolutional Networks for Visual Recognition and Description](https://arxiv.org/abs/1411.4389)
###   Other Versions of Image Captioning
-   [DenseCap](https://arxiv.org/abs/1511.07571)
-   [Deep Visual-Semantic Alignments for Generating Image Descriptions](https://arxiv.org/abs/1412.2306)
-   [Nocaps](https://arxiv.org/abs/1812.08658)
-   [Deep Compositional Captioning](https://arxiv.org/abs/1511.05284)
-   [Captioning Images with Diverse Objects](https://arxiv.org/abs/1606.07770)
###   Incorporating Visual Attributes
-   [Boosting Image Captioning with Attributes](https://arxiv.org/abs/1611.01646)
-   [Image Captioning with Semantic Attention](https://arxiv.org/abs/1603.03925)
-   [Show, Observe, and Tell](https://www.ijcai.org/proceedings/2018/0084.pdf)
###   Decoding Methods
-   [Beam Search](https://en.wikipedia.org/wiki/Beam_search)
-   [Guided Open Vocabulary Image Captioning with Constrained Beam Search](https://arxiv.org/abs/1612.00576)
-   [Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search](https://arxiv.org/abs/1704.07138)
###   Style Transfer
-   [Show, Adapt, and Tell](https://arxiv.org/abs/1705.00930)
-   [StyleNet](https://www.microsoft.com/en-us/research/uploads/prod/2017/06/Generating-Attractive-Visual-Captions-with-Styles.pdf)
###   Discriminability
-   [Discriminability Objective for Training Descriptive Captions](https://arxiv.org/abs/1803.04376)
-   [Show, Tell, and Discriminate](https://arxiv.org/abs/1803.08314)
###   Navigation
-   [Vision-and-Language Navigation](https://arxiv.org/abs/1711.07280)
###   Open-Source Pretrained Models
-   [Pythia](https://github.com/facebookresearch/pythia)
###   Field Leaders
-   [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/)
-   [Devi Parikh](https://www.cc.gatech.edu/~parikh/)
-   [Dhruv Batra](https://www.cc.gatech.edu/~dbatra/)
-   [Peter Anderson](https://panderson.me/)
-   [Stefen Lee](http://web.engr.oregonstate.edu/~leestef/)
## Miscellaneous
###   Augmentation
-   [Mixup](https://arxiv.org/abs/1710.09412)
-   [AutoAugment](https://arxiv.org/abs/1805.09501)
-   [PBA](https://arxiv.org/abs/1905.05393)
-   [RandAugment](https://arxiv.org/abs/1909.13719)
###   Other
-   [Neural ODEs](https://arxiv.org/abs/1806.07366)
