# ML Reading List
Welcome to the Machine Learning at Berkeley reading list! This was assembled by students at UC Berkeley, and was designed to be a good reference for those in the intermediate stages of learning ML.

Made By: **Chris Bender** (chris at ml dot berkeley dot edu) and **Phillip Kravtsov** (phillipk at ml dot berkeley dot edu)

For various sub-topics in machine learning, we have assembled a hierarchical reading list, alongside an introduction to the sub-topic. For each paper, we have put a star-rating, with (\*) denoting low-importance papers that are only helpful for those most interested in the topic, (\*\*) for papers with medium importance, and (\*\*\*) for the high-importance papers that should be read first.

**Beginning Guide**

The following papers give you a flavor of each of the sections, and don’t require much extra knowledge beyond basic deep learning concepts (you should know about MLPs/CNNs and how to train them).

-   [EfficientNet](https://arxiv.org/abs/1905.11946)

-   When we want to scale up a neural network to a bigger task, the most natural way to do so is to add more layers. However, we also have control over layer width and the resolution of our input (we usually downsize ImageNet samples to 224x224 resolution, but you can increase the resolution for a minor accuracy boost). In practice, though, it is not clear how to scale up these three components; the EfficientNet paper gives a more principled method for model scaling that achieves significantly better accuracy-FLOP tradeoff.

-   [Yolo](https://arxiv.org/abs/1506.02640)

-   Yolo is one of the more important (and easier to understand) methods for detecting objects within an image.

-   [PointNet](https://arxiv.org/abs/1612.00593)

-   We usually see computer vision problems operate on 2D (image) data, but we sometimes want to do processing on 3D data. (One predominant example of this is 3D object detection from LiDAR points for self-driving cars.) This paper proposes a simple method for processing 3D point clouds and serves as the backbone for many more sophisticated 3D object detectors.

-   [Show and Tell](https://arxiv.org/abs/1411.4555)

-   This paper gives a simple LSTM-based architecture that can automatically generate a caption for an image. A great intro that puts together well-understood pieces to solve an interesting new problem.
-   In order to fully understand the paper, it is helpful to have familiarity with the evaluation measure that they use (the BLEU score). [Here](https://en.wikipedia.org/wiki/BLEU) is the Wikipedia page for the BLEU score and is a good start.

-   [Transformer](https://arxiv.org/abs/1706.03762)

-   Many of the recent NLP models (e.g., BERT) rely heavily on the 'self-attention' operation. The original Transformer paper helped start this trend and is a great resource for understanding how self-attention works.

-   [Distributed Overview](https://arxiv.org/abs/1802.09941)

-   Much of the recent progress in machine learning has been fueled by the increasing ability to train larger neural networks over many different GPUs (and perhaps even on multiple machines). This problem of distributed training is one of the fundamental issues that systems researchers study (among others). This paper provides a great overview of the different systems problems that are faced in machine learning.

-   [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)

-   When deploying ML systems in the real world, there are lots of unexpected challenges that you face (data dependencies, feedback loops, overlapping data pipelines, etc.). Learn about the practical side of ML deployment, written by a bunch of Googlers.

-   [Rethinking Generalization](https://arxiv.org/abs/1611.03530)

-   In machine learning, we oftentimes talk about the problem of_generalization_: how do we ensure that neural networks perform correctly on inputs that have not been seen before. This paper casts doubt on the ability for all known classical techniques (such as VC Dimension) to explain why neural networks generalize, and claims that we need better fundamental understanding of the generalization properties of neural networks.
-   Although not strictly necessary, it may be helpful to understand VC Dimension before reading the paper. [Here](https://towardsdatascience.com/measuring-the-power-of-a-classifier-c765a7446c1c) is a reasonably good blog post that introduces the VC Dimension.

-   [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)

-   A very cool paper that points out an interesting property of neural networks: Given a randomly initialized neural network, it is usually possible to identify a smaller sub-network that can train just as easily as the original, larger network. The paper hypothesizes that this property can help explain why larger networks usually give better performance.

-   [AutoAugment](https://arxiv.org/abs/1805.09501)

-   We usually augment data by applying random transformations (crops, brightness/contrast shifts, etc.). This paper shows that we can improve this by having a neural network choose the augmentation strategy instead.
-   Fully understanding this paper will require a bit of reinforcement learning knowledge: In particular, they use Proximal Policy Optimization (PPO) to train the augmentation strategy. However, a basic familiarlity with reinforcement learning will be sufficient to get the main idea and understand the results.

-   [CycleGAN](https://arxiv.org/abs/1703.10593)

-   Make pretty pictures and transform horses into zebras with GANs! This paper uses a cool trick (cycle-consistency) to make this possible without collecting a "horse-to-zebra" dataset.

-   [PixelCNN](https://arxiv.org/abs/1606.05328)

-   Intuitively, the "ultimate" form of unsupervised learning is to learn an exact probability density function over your entire data space: This allows you to both estimate the likelihood of input samples, and to generate new samples from the distribution. This task of estimating the exact density is termed_likelihood modeling_. The PixelCNN is one very important approach to do likelihood modeling on images directly.

-   [VAE Tutorial](https://arxiv.org/abs/1606.05908)

-   One drawback of many unsupervised learning models is that they are usually good at one thing only: For example, GANs can produce really high-quality samples, but are not helpful in estimating the probability of an image. Similarly, PixelCNNs are great for estimating image probabilities, but are not great for producing samples. In contrast, variational autoencoders (VAEs) can produce samples, estimate approximate likelihoods, and interpolate between images, all in one model.

-   [RotNet](https://arxiv.org/abs/1803.07728)

-   We know that we can use the hidden layers of a pretrained classifier to extract meaningful features from an image (i.e., features that are more sensitive to semantic changes in the image), but this implicitly relies on labels (the original pretrained classifier was trained with labels). This paper proposes a method to learn these representations with no human-labeled images at all.

-   [MAML](https://arxiv.org/abs/1703.03400)

-   Humans are much more efficient than neural networks: We only need a few examples (perhaps even one) of a given class to be able to reliably recognize it. One explanation for this is that we have "learned how to learn"; that is, we have seen many other objects in the real world, so we have an understanding of the general properties of objects. The study of "learning to learn" is termed _meta-learning_, and the MAML paper introduces a very simple approach for meta-learning: Just train the model to be easy to fine-tune.

**Good Resources**

Arxiv Sanity ([www.arxiv-sanity.com](http://www.arxiv-sanity.com/)). Developed by Andrej Karpathy, Arxiv Sanity is a great resource for finding the most relevant papers published on Arxiv. The website has tools for saving papers to a personal library, seeing recommendations based on your saved papers, and filtering papers by most-saved or most-discussed.

Arxiv Vanity ([www.arxiv-vanity.com](http://www.arxiv-vanity.com/)). Arxiv Vanity renders Arxiv PDFs in a mobile-friendly HTML format.

Depth First Learning ([www.depthfirstlearning.com](https://www.depthfirstlearning.com/)). Typically tackling papers that require more background knowledge, DFL is a great resource for very high-quality explanations of ML research concepts.

Distill.pub ([distill.pub](https://distill.pub/)). Typically tackling papers that require more background knowledge, DFL is a great resource for very high-quality blogs and explanations of ML research concepts.

Berkeley AI Research Blog ([www.bair.berkeley.edu/blog](https://bair.berkeley.edu/blog/)). A great resource for high-quality blogs written by Berkeley ML researchers.

r/MachineLearning ([www.reddit.com/r/machinelearning](https://www.reddit.com/r/machinelearning/)). Reddit community for discussions on ML.

**Twitter Accounts**

-   ML Researchers

-   Andrej Karpathy ([@karpathy](https://twitter.com/karpathy/))
-   Woj Zaremba ([@woj\_zaremba](https://twitter.com/woj_zaremba/))
-   Oriol Vinayls ([@OriolVinyalsML](https://twitter.com/OriolVinyalsML/))
-   David Ha ([@hardmaru](https://twitter.com/hardmaru/))
-   François Chollet ([@fchollet](https://twitter.com/fchollet/))
-   Justin Johnson ([@jcjohnss](https://twitter.com/jcjohnss/))
-   Jeff Dean ([@JeffDean](https://twitter.com/JeffDean/))
-   Pieter Abbeel ([@pabbeel](https://twitter.com/pabbeel/))

-   Research Labs

-   DeepMind ([@DeepMindAI](https://twitter.com/DeepMindAI/))
-   OpenAI ([@OpenAI](https://twitter.com/OpenAI/))
-   Berkeley AI Research ([@berkeley\_ai](https://twitter.com/berkeley_ai/))
-   Stanford NLP Group ([@stanfordnlp](https://twitter.com/stanfordnlp/))

**Other Reading Lists**

-   Chelsea Finn and Sergey Levine: [ICML 2019 Meta-Learning Tutorial](https://sites.google.com/view/icml19metalearning)
