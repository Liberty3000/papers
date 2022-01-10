Week 1 - 1/7/22

### [Thinking Like Transformers](https://arxiv.org/abs/2106.06981) ![arXiv](https://img.shields.io/badge/arXiv-2106.06981-maroon)

  - What is the computational model behind a Transformer?

    - Token-by-token processing of RNNs can be conceptualized as finite state automata.

    - RNNs -> finite state machines, Transformers -> ???

  - **This paper proposes a computational model for the transformer-encoder in the form of a programming language.**

    - The Restricted Access Sequence Processing Language (RASP) is built from mapping Transformer-Encoder components  (attention and feed-forward computations) into simple primitives.

    - The model can be used to relate task difficulty in terms of the number of required layers and attention heads

      - Analyzing a RASP program implies a maximum number of heads and layers necessary to encode a task in a transformer.

    - The RASP language captures the unique information-flow constraints under which a transformer operates as it processes input sequences.

    - The model helps reason about Transformer operates at a higher-level of abstraction, reasoning in terms of a  composition of sequence operations rather than neural network primitives.

  - Previous work on Transformers explore their computational power, but does not provide a computational model

   - Considering computation problems and their implementations in RASP allows us to “think like a transformer” while abstracting away the technical details of a neural network in favor of symbolic programs.

   - A compiled RASP program can indeed be realized in a neural transformer and occasionally is even the solution found by a transformer trained on the task using gradient descent

   - The iterative process of a transformer is then not along the length of the input sequence but rather the depth of the computation: the number of layers it applies to its input as it works towards its final result.

   - A RASP computation over length-n input involves manipulation of sequences of length n, and matrices of size n×n. There are no sequences or matrices of different sizes in a RASP computation.

  - The model can use an additional auxiliary loss for attention supervision, given the MSE of the difference between its attention heatmaps and those expected by the RASP solution.


### [Transformer for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) ![arXiv](https://img.shields.io/badge/arXiv-2010.11929-maroon)

  - **This paper shows that the reliance in CNNs in computer vision applications is not necessary and a pure Transformer applied directly to sequences of image patches can perform very well on image classification tasks.**

  - The Vision Transformer (ViT) attains excellent results compared to SOTA CNNs while requiring substantially fewer computational resources to train.

    - The authors attempt to accomplish this by applying a Transformer architecture directly to images, with the fewest possible modifications.

    - An advantage of this intentionally simple setup is that scalable NLP Transformer architecture and their efficient implementations can be used almost out-of-the-box.

    - Additionally, the ViT architecture can handle arbitrary sequence lengths.

  - When trained on mid-sized datasets such as ImageNet without strong regularization, these models yield modest accuracies of a few percentage points below ResNets of comparable size.
    - The picture changes if the ViT models are trained on larger datasets (14M - 300M).
    - Large-scale pre-training trumps inductive bias (leveraged by CNNs).

  - Large Transformer-based models are often pre-trained on a large corpora and then fine-tuned on a smaller, task-specific dataset.
    - BERT uses a denoising self-supervised pre-training task.
    - GPT models use language modeling as the pre-training task.

  - The most recent related work is the model in ![arXiv](https://img.shields.io/badge/arXiv-1911.03584-maroon), which extracts patches of size 2x2 from the input image and applies full self-attention on top.

  - To handle 2D images, an image of shape HxWxC is reshaped into a sequence of flattened 2D patches with shape Nx(P^2xC), where HxW is the original resolution, C is the number of channels and PxP is the resolution of each image patch, and N=HxW/(P^2) is the number of image patches, which also serves as the effective input sequence length.

  - For the inductive bias, in ViT, only MLP layers are local and translationally equivariant, while the self-attention layers are global.

  - The authors found it beneficial to fine-tune the model at a higher resolution than pre-training.
    - When feeding images of higher resolution, they keep the patch size the same, which results in a larger effective sequence length.
    - In this case, the pre-trained position embeddings may no longer be meaningful, so they perform 2D interpolation of the pre-trained position embeddings.

  - From experiments, the authors find that ViTs generally outperform ResNets with the same computational budget.
    - Hybrid models improve upon pure Transformers for smaller model sizes, but the gap vanishes for larger models.
    - ViTs overfit more than ResNets with comparable computational cost on smaller datasets.
    - These results reinforce the intuition that the convolutional inductive bias is useful for smaller datasets, but for larger ones, learning the relevant patterns directly from the data is sufficient, if not more beneficial.

  - Self-attention allows ViTs to integrate information across the entire image, even in the lowest layers.


### [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841) ![arXiv](https://img.shields.io/badge/arXiv-2106.06981-maroon)

  - This paper demonstrates the effectiveness of combining the inductive bias of CNNs with the expressivity of Transformers.
    - (1) CNNs learn the context-rich vocabulary of image constituents.
    - (2) Transformers efficiently model their composition within high-resolution images.
    - Non-spatial information, such as object classes, and spatial information, such as segmentations can be used to control the generated image.

  - Problem: the increased expressivity of Transformers come with a quadratic increase in computation, because all pairwise interactions (pixels, patches, feature maps, etc.) are taken into account.
  - CNNs exhibit a strong biases towards spatial locality and spatial invariance.
    - Often ineffective if a more holistic understanding of an input image is required.

  - Transformer consistently outperform their convolutional counterparts for low-resolution images.

  - CNNs exploit structure of images by restricting interactions between input variables o a local neighborhood defined by the kernel size of the convolutional kernel.

  - The authors use an adversarial approach to ensure the learned dictionary of image parts by their method captures perceptually important local structure to alleviate the need for modeling low-level statistics with the Transformer architecture.

    - To aggregate context from everywhere within an image, a single attention layer is applied on the lowest resolution.

  - The defining characteristic of the Transformer architecture is that it models interactions between its inputs solely through attention.

  - The authors employ a two-stage approach; first learns and encoding of the data then learns a probabilistic model of this encoding.

    - This paper demonstrates that a powerful first stage, which captures as much context as possible in the learned representation is critical in enabling high-resolution synthesis.

    - The authors use a convolutional VQGAN to learn a codebook of visual parts, whose composition is subsequently modeled with an autoregressive transformer architecture. The VQGAN is a variation of the VQVAE from ![arXiv](https://img.shields.io/badge/arXiv-1711.00937-maroon), with an additional use of a patch-based discriminator model and a perceptual loss.

    - A discrete codebook is the interface between these architecture and a patch-based discriminator enables strong compression of the codebook while retaining high perceptual quality.

  - The codebook is discrete; therefore, it is not trivially differentiable. To overcome this, the authors backpropagate through the non-differentiable quantization operation by using a straight-through gradient estimator, which simply copies gradients from the decoder to the encoder.

  - The model is trained end-to-end using a reconstruction loss and the "commitment loss" from ![arXiv](https://img.shields.io/badge/arXiv-1711.00937-maroon).

  - To generate images in the megapixel regime, training requires patch-wise and cropped images to restrict the length of the input to a maximally feasible size during training. To sample images, a Transformer is employed in a sliding-window manner.
