# Exercises

- [Chapter 14: Deep Computer Vision Using Convolutional Neural Networks](#c14)
- [Chapter 15: Processing Sequences Using RNNs and CNNs](#c15)
- [Chapter 16: Natural Language Processing with RNNs and Attention](#c16)

<a name="c14"></a>
## Chapter 14: Deep Computer Vision Using Convolutional Neural Networks

1. **What are the advantages of a CNN over a fully connected DNN for image classification?**
    <details>
    <summary>Answer</summary>
    These are the main advantages of a CNN over a fully connected DNN for image classification:

    • Because consecutive layers are only partially connected and because it heavily reuses its weights,
    a CNN has many fewer parameters than a fully connected DNN, which makes it much faster to train,
    reduces the risk of overfitting, and requires much less training data.

    • When a CNN has learned a
    kernel that can detect a particular feature, it can detect that feature anywhere in the image. In
    contrast, when a DNN learns a feature in one location, it can detect it only in that particular
    location. Since images typically have very repetitive features, CNNs are able to generalize much
    better than DNNs for image processing tasks such as classification, using fewer training examples.

    • Finally, a DNN has no prior knowledge of how pixels are organized; it does not know that nearby
    pixels are close. A CNN’s architecture embeds this prior knowledge. Lower layers typically identify
    features in small areas of the images, while higher layers combine the lower-level features into
    larger features. This works well with most natural images, giving CNNs a decisive head start compared
    to DNNs.

    ```python
    import tensorflow as tf
    ```
    </details>

2. **Consider a CNN composed of three convolutional layers, each with 3 × 3 kernels, a stride of 2,
   and "same" padding. The lowest layer outputs 100 feature maps, the middle one outputs 200, and
   the top one outputs 400. The input images are RGB images of 200 × 300 pixels.**

    <details>
    <summary>Answer</summary>
    Let’s compute how many parameters the CNN has. Since its first convolutional layer has 3 × 3
    kernels, and the input has three channels (red, green, and blue), each feature map has 3 × 3 × 3
    weights, plus a bias term. That’s 28 parameters per feature map. Since this first convolutional
    layer has 100 feature maps, it has a total of 2,800 parameters. The second convolutional layer has 3
    × 3 kernels and its input is the set of 100 feature maps of the previous layer, so each feature map
    has 3 × 3 × 100 = 900 weights, plus a bias term. Since it has 200 feature maps, this layer has 901 ×
    200 = 180,200 parameters. Finally, the third and last convolutional layer also has 3 × 3 kernels,
    and its input is the set of 200 feature maps of the previous layers, so each feature map has 3 × 3 ×
    200 = 1,800 weights, plus a bias term. Since it has 400 feature maps, this layer has a total of
    1,801 × 400 = 720,400 parameters. All in all, the CNN has 2,800 + 180,200 + 720,400 = 903,400
    parameters.

    Now let’s compute how much RAM this neural network will require (at least) when making a prediction
    for a single instance. First let’s compute the feature map size for each layer. Since we are using a
    stride of 2 and "same" padding, the horizontal and vertical dimensions of the feature maps are
    divided by 2 at each layer (rounding up if necessary). So, as the input channels are 200 × 300
    pixels, the first layer’s feature maps are 100 × 150, the second layer’s feature maps are 50 × 75,
    and the third layer’s feature maps are 25 × 38. Since 32 bits is 4 bytes and the first convolutional
    layer has 100 feature maps, this first layer takes up 4 × 100 × 150 × 100 = 6 million bytes (6 MB).
    The second layer takes up 4 × 50 × 75 × 200 = 3 million bytes (3 MB). Finally, the third layer takes
    up 4 × 25 × 38 × 400 = 1,520,000 bytes (about 1.5 MB). However, once a layer has been computed, the
    memory occupied by the previous layer can be released, so if everything is well optimized, only 6 +
    3 = 9 million bytes (9 MB) of RAM will be required (when the second layer has just been computed,
    but the memory occupied by the first layer has not been released yet). But wait, you also need to
    add the memory occupied by the CNN’s parameters! We computed earlier that it has 903,400 parameters,
    each using up 4 bytes, so this adds 3,613,600 bytes (about 3.6 MB). The total RAM required is
    therefore (at least) 12,613,600 bytes (about 12.6 MB).

    Lastly, let’s compute the minimum amount of RAM required when training the CNN on a mini-batch of 50
    images. During training TensorFlow uses backpropagation, which requires keeping all values
    computed during the forward pass until the reverse pass begins. So we must compute the total RAM
    required by all layers for a single instance and multiply that by 50. At this point, let’s start
    counting in megabytes rather than bytes. We computed before that the three layers require
    respectively 6, 3, and 1.5 MB for each instance. That’s a total of 10.5 MB per instance, so for 50
    instances the total RAM required is 525 MB. Add to that the RAM required by the input images, which
    is 50 × 4 × 200 × 300 × 3 = 36 million bytes (36 MB), plus the RAM required for the model
    parameters, which is about 3.6 MB (computed earlier), plus some RAM for the gradients (we will
    neglect this since it can be released gradually as backpropagation goes down the layers during the
    reverse pass). We are up to a total of roughly 525 + 36 + 3.6 = 564.6 MB, and that’s really an
    optimistic bare minimum.
    </details>

3. **If your GPU runs out of memory while training a CNN, what are five things you could try to solve the problem?**

    <details>
    <summary>Answer</summary>
    If your GPU runs out of memory while training a CNN, here are five things you could try to solve the
    problem (other than purchasing a GPU with more RAM):

    • Reduce the mini-batch size.

    • Reduce dimensionality using a larger stride in one or more layers.

    • Remove one or more layers.

    • Use 16-bit floats instead of 32-bit floats.

    • Distribute the CNN across multiple devices.
    </details>

4. **Why would you want to add a max pooling layer rather than a convolutional layer with the same stride?**

    <details>
    <summary>Answer</summary>
    A max pooling layer has no parameters at all, whereas a convolutional layer has quite a few (see the
    previous questions).
    </details>

5. **When would you want to add a local response normalization layer?**

    <details>
    <summary>Answer</summary>
    A local response normalization layer makes the neurons that most strongly activate inhibit neurons
    at the same location but in neighboring feature maps, which encourages different feature maps to
    specialize and pushes them apart, forcing them to explore a wider range of features. It is typically
    used in the lower layers to have a larger pool of low-level features that the upper layers can build
    upon.
    </details>

6. **Can you name the main innovations in AlexNet, compared to LeNet-5? What about the main innovations in GoogLeNet, ResNet, SENet, and Xception?**

    <details>
    <summary>Answer</summary>
    The main innovations in AlexNet compared to LeNet-5 are that it is much larger and deeper, and it
    stacks convolutional layers directly on top of each other, instead of stacking a pooling layer on
    top of each convolutional layer. The main innovation in GoogLeNet is the introduction of inception
    modules, which make it possible to have a much deeper net than previous CNN architectures, with
    fewer parameters. ResNet’s main innovation is the introduction of skip connections, which make it
    possible to go well beyond 100 layers. Arguably, its simplicity and consistency are also rather
    innovative. SENet’s main innovation was the idea of using an SE block (a two-layer dense network)
    after every inception module in an inception network or every residual unit in a ResNet to
    recalibrate the relative importance of feature maps. Finally, Xception’s main innovation was the use
    of depthwise separable convolutional layers, which look at spatial patterns and depthwise patterns
    separately.
    </details>

7. **What is a fully convolutional network? How can you convert a dense layer into a convolutional layer?**

    <details>
    <summary>Answer</summary>
    Fully convolutional networks are neural networks composed exclusively of convolutional and pooling
    layers. FCNs can efficiently process images of any width and height (at least above the minimum
    size). They are most useful for object detection and semantic segmentation because they only need to
    look at the image once (instead of having to run a CNN multiple times on different parts of the
    image). If you have a CNN with some dense layers on top, you can convert these dense layers to
    convolutional layers to create an FCN: just replace the lowest dense layer with a convolutional
    layer with a kernel size equal to the layer’s input size, with one filter per neuron in the dense
    layer, and using "valid" padding. Generally the stride should be 1, but you can set it to a higher
    value if you want. The activation function should be the same as the dense layer’s. The other dense
    layers should be converted the same way, but using 1 × 1 filters. It is actually possible to convert
    a trained CNN this way by appropriately reshaping the dense layers’ weight matrices.
    </details>

8. **What is the main technical difficulty of semantic segmentation?**

    <details>
    <summary>Answer</summary>
    The main technical difficulty of semantic segmentation is the fact that a lot of the spatial
    information gets lost in a CNN as the signal flows through each layer, especially in pooling layers
    and layers with a stride greater than 1. This spatial information needs to be restored somehow to
    accurately predict the class of each pixel.
    </details>

9. **Build your own CNN from scratch and try to achieve the highest possible accuracy on MNIST.**

    _See notebooks_

10. **Use transfer learning for large image classification, going through these steps:<br>**

    _See notebooks_

    a. Create a training set containing at least 100 images per class. For example, you could
    classify your own pictures based on the location (beach, mountain, city, etc.), or alternatively
    you can use an existing dataset (e.g., from TensorFlow Datasets).

    b. Split it into a training set, a validation set, and a test set.

    c. Build the input pipeline, including the appropriate preprocessing operations, and optionally
    add data augmentation.

    d. Fine-tune a pre-trained model on this dataset.

11. **Go through TensorFlow’s Style Transfer tutorial. It is a fun way to generate art
using Deep Learning.**

    _See notebooks_

<a name="c15"></a>
## Chapter 15: Processing Sequences Using RNNs and CNNs

1. **Can you think of a few applications for a sequence-to-sequence RNN? What about a
   sequence-to-vector RNN, and a vector-to-sequence RNN?**

    <details>
    <summary>Answer</summary>
    Here are a few RNN applications:
    • For a sequence-to-sequence RNN: predicting the weather (or any other time series), machine
    translation (using an Encoder–Decoder architecture), video captioning, speech to text, music
    generation (or other sequence generation), identifying the chords of a song
    • For a sequence-to-vector RNN: classifying music samples by music genre, analysing the
    sentiment of a book review, predicting what word an aphasic patient is thinking of based on
    readings from brain implants, predicting the probability that a user will want to watch a
    movie based on their watch history (this is one of many possible implementations of
    collaborative filtering for a recommender system)
    • For a vector-to-sequence RNN: image captioning, creating a music playlist based on an
    embedding of the current artist, generating a melody based on a set of parameters, locating
    pedestrians in a picture (e.g., a video frame from a self-driving car’s camera)
    </details>

2. **How many dimensions must the inputs of an RNN layer have? What does each dimension represent?
   What about its outputs?**

    <details>
    <summary>Answer</summary>
    An RNN layer must have three-dimensional inputs: the first dimension is the batch dimension (its
    size is the batch size), the second dimension represents the time (its size is the number of time
    steps), and the third dimension holds the inputs at each time step (its size is the number of input
    features per time step). For example, if you want to process a batch containing 5 time series of 10
    time steps each, with 2 values per time step (e.g., the temperature and the wind speed), the shape
    will be [5, 10, 2]. The outputs are also three-dimensional, with the same first two dimensions, but
    the last dimension is equal to the number of neurons. For example, if an RNN layer with 32 neurons
    processes the batch we just discussed, the output will have a shape of [5, 10, 32].
    </details>

3. **If you want to build a deep sequence-to-sequence RNN, which RNN layers should have
   `return_sequences=True`? What about a sequence-to-vector RNN?**

    <details>
    <summary>Answer</summary>
    To build a deep sequence-to-sequence RNN using Keras, you must set `return_sequences=True` for all RNN
    layers. To build a sequence-to-vector RNN, you must set `return_sequences=True` for all RNN layers
    except for the top RNN layer, which must have `return_sequences=False` (or do not set this argument at
    all, since False is the default).
    </details>

4. **Suppose you have a daily univariate time series, and you want to forecast the next seven days.
   Which RNN architecture should you use?**

    <details>
    <summary>Answer</summary>
    If you have a daily univariate time series, and you want to forecast the next seven days, the
    simplest RNN architecture you can use is a stack of RNN layers (all with `return_sequences=True`
    except for the top RNN layer), using seven neurons in the output RNN layer. You can then train this
    model using random windows from the time series (e.g., sequences of 30 consecutive days as the
    inputs, and a vector containing the values of the next 7 days as the target). This is a sequence-
    to-vector RNN. Alternatively, you could set `return_sequences=True` for all RNN layers to create a
    sequence-to-sequence RNN. You can train this model using random windows from the time series, with
    sequences of the same length as the inputs as the targets. Each target sequence should have seven
    values per time step (e.g., for time step t, the target should be a vector containing the values at
    time steps t + 1 to t + 7).
    </details>

5. **What are the main difficulties when training RNNs? How can you handle them?**

    <details>
    <summary>Answer</summary>
    The two main difficulties when training RNNs are unstable gradients (exploding or vanishing) and a
    very limited short-term memory. These problems both get worse when dealing with long sequences. To
    alleviate the unstable gradients problem, you can use a smaller learning rate, use a saturating
    activation function such as the hyperbolic tangent (which is the default), and possibly use gradient
    clipping, Layer Normalization, or dropout at each time step. To tackle the limited short-term memory
    problem, you can use LSTM or GRU layers (this also helps with the unstable gradients problem).
    </details>

6. **Can you sketch the LSTM cell’s architecture?**

    <details>
    <summary>Answer</summary>
    An LSTM cell’s architecture looks complicated, but it’s actually not too hard if you understand the
    underlying logic. The cell has a short-term state vector and a long-term state vector. At each time
    step, the inputs and the previous short-term state are fed to a simple RNN cell and three gates: the
    forget gate decides what to remove from the long-term state, the input gate decides which part of
    the output of the simple RNN cell should be added to the long-term state, and the output gate
    decides which part of the long-term state should be output at this time step (after going through
    the *_tanh_* activation function). The new short-term state is equal to the output of the cell. See
    Figure 15-9.
    </details>

7. **Why would you want to use 1D convolutional layers in an RNN?**

    <details>
    <summary>Answer</summary>
    An RNN layer is fundamentally sequential: in order to compute the outputs at time step t, it has to
    first compute the outputs at all earlier time steps. This makes it impossible to parallelize. On the
    other hand, a 1D convolutional layer lends itself well to parallelization since it does not hold a
    state between time steps. In other words, it has no memory: the output at any time step can be
    computed based only on a small window of values from the inputs without having to know all the past
    values. Moreover, since a 1D convolutional layer is not recurrent, it suffers less from unstable
    gradients. One or more 1D convolutional layers can be useful in an RNN to efficiently preprocess the
    inputs, for example to reduce their temporal resolution (downsampling) and thereby help the RNN
    layers detect long-term patterns. In fact, it is possible to use only convolutional layers, for
    example by building a WaveNet architecture.
    </details>

8. **Which neural network architecture could you use to classify videos?**

    <details>
    <summary>Answer</summary>
    To classify videos based on their visual content, one possible architecture could be to take (say)
    one frame per second, then run every frame through the same convolutional neural network (e.g., a
    pretrained Xception model, possibly frozen if your dataset is not large), feed the sequence of
    outputs from the CNN to a sequence-to-vector RNN, and finally run its output through a *_softmax_*
    layer, giving you all the class probabilities. For training you would use cross entropy as the
    cost function. If you wanted to use the audio for classification as well, you could use a stack of
    strided 1D convolutional layers to reduce the temporal resolution from thousands of audio frames per
    second to just one per second (to match the number of images per second), and concatenate the output
    sequence to the inputs of the sequence-to-vector RNN (along the last dimension).
    </details>

9. **Train a classification model for the SketchRNN dataset, available in TensorFlow Datasets.**

    _See notebooks_

10. **Download the Bach chorales dataset and unzip it. It is composed of 382 chorales composed by
    Johann Sebastian Bach. Each chorale is 100 to 640 time steps long, and each time step contains 4
    integers, where each integer corresponds to a note’s index on a piano (except for the value 0,
    which means that no note is played). Train a model—recurrent, convolutional, or both—that can
    predict the next time step (four notes), given a sequence of time steps from a chorale. Then use
    this Exercises | 523 model to generate Bach-like music, one note at a time: you can do this by
    giving the model the start of a chorale and asking it to predict the next time step, then
    appending these time steps to the input sequence and asking the model for the next note, and so
    on. Also make sure to check out Google’s Coconet model, which was used for a nice Google doodle
    about Bach.**

    _See notebooks_

<a name="c16"></a>
# Chapter 16: Natural Language Processing with RNNs and Attention

1. **What are the pros and cons of using a stateful RNN versus a stateless RNN?**

    <details>
    <summary>Answer</summary>
    Stateless RNNs can only capture patterns whose length is less than, or equal to, the size of the
    windows the RNN is trained on. Conversely, stateful RNNs can capture longer-term patterns. However,
    implementing a stateful RNN is much harder—especially preparing the dataset properly. Moreover,
    stateful RNNs do not always work better, in part because consecutive batches are not independent and
    identically distributed (IID). Gradient Descent is not fond of non-IID datasets.
    </details>

2. **Why do people use Encoder–Decoder RNNs rather than plain sequence-to- sequence RNNs for automatic
   translation?**

    <details>
    <summary>Answer</summary>
    In general, if you translate a sentence one word at a time, the result will be terrible. For
    example, the French sentence “Je vous en prie” means “You are welcome,” but if you translate it one
    word at a time, you get “I you in pray.” Huh? It is much better to read the whole sentence first and
    then translate it. A plain sequence-to- sequence RNN would start translating a sentence immediately
    after reading the first word, while an Encoder–Decoder RNN will first read the whole sentence and
    then translate it. That said, one could imagine a plain sequence-to-sequence RNN that would output
    silence whenever it is unsure about what to say next (just like human translators do when they must
    translate a live broadcast).
    </details>

3. **How can you deal with variable-length input sequences? What about variable- length output
   sequences?**

    <details>
    <summary>Answer</summary>
    Variable-length input sequences can be handled by padding the shorter sequences so that all
    sequences in a batch have the same length, and using masking to ensure the RNN ignores the padding
    token. For better performance, you may also want to create batches containing sequences of similar
    sizes. Ragged tensors can hold sequences of variable lengths, and `tf.keras` will likely support them
    eventually, which will greatly simplify handling variable-length input sequences (at the time of
    this writing, it is not the case yet). Regarding variable-length output sequences, if the length of
    the output sequence is known in advance (e.g., if you know that it is the same as the input
    sequence), then you just need to configure the loss function so that it ignores tokens that come
    after the end of the sequence. Similarly, the code that will use the model should ignore tokens
    beyond the end of the sequence. But generally the length of the output sequence is not known ahead
    of time, so the solution is to train the model so that it outputs an end-of- sequence token at the
    end of each sequence.
    </details>

4. **What is beam search and why would you use it? What tool can you use to implement it?**

    <details>
    <summary>Answer</summary>
    Beam search is a technique used to improve the performance of a trained Encoder–Decoder model, for
    example in a neural machine translation system. The algorithm keeps track of a short list of the k
    most promising output sentences (say, the top three), and at each decoder step it tries to extend
    them by one word; then it keeps only the k most likely sentences. The parameter k is called the beam
    width: the larger it is, the more CPU and RAM will be used, but also the more accurate the system
    will be. Instead of greedily choosing the most likely next word at each step to extend a single
    sentence, this technique allows the system to explore several promising sentences simultaneously.
    Moreover, this technique lends itself well to parallelization. You can implement beam search
    fairly easily using TensorFlow Addons.
    </details>

5. **What is an attention mechanism? How does it help?**

    <details>
    <summary>Answer</summary>
    An attention mechanism is a technique initially used in Encoder–Decoder models to give the decoder
    more direct access to the input sequence, allowing it to deal with longer input sequences. At each
    decoder time step, the current decoder’s state and the full output of the encoder are processed by
    an alignment model that outputs an alignment score for each input time step. This score indicates
    which part of the input is most relevant to the current decoder time step. The weighted sum of the
    encoder output (weighted by their alignment score) is then fed to the decoder, which produces the
    next decoder state and the output for this time step. The main benefit of using an attention
    mechanism is the fact that the Encoder–Decoder model can successfully process longer input
    sequences. Another benefit is that the alignment scores makes the model easier to debug and
    interpret: for example, if the model makes a mistake, you can look at which part of the input it was
    paying attention to, and this can help diagnose the issue. An attention mechanism is also at the
    core of the Transformer architecture, in the Multi-Head Attention layers. See the next answer.
    </details>

6. **What is the most important layer in the Transformer architecture? What is its purpose?**

    <details>
    <summary>Answer</summary>
    The most important layer in the Transformer architecture is the Multi-Head Attention layer (the
    original Transformer architecture contains 18 of them, including 6 Masked Multi-Head Attention
    layers). It is at the core of language models such as BERT and GPT-2. Its purpose is to allow the
    model to identify which words are most aligned with each other, and then improve each word’s
    representation using these contextual clues.
    </details>

7. **When would you need to use sampled softmax?**

    <details>
    <summary>Answer</summary>
    Sampled softmax is used when training a classification model when there are many classes (e.g.,
    thousands). It computes an approximation of the cross- entropy loss based on the logit predicted by
    the model for the correct class, and the predicted logits for a sample of incorrect words. This
    speeds up training considerably compared to computing the softmax over all logits and then
    estimating the cross-entropy loss. After training, the model can be used normally, using the regular
    softmax function to compute all the class probabilities based on all the logits.
    </details>

8. **Embedded Reber grammars were used by Hochreiter and Schmidhuber in their paper about LSTMs. They
   are artificial grammars that produce strings such as “BPBTSXXVPSEPE.” Check out Jenny Orr’s nice
   introduction to this topic. Choose a particular embedded Reber grammar (such as the one
   represented on Jenny Orr’s page), then train an RNN to identify whether a string respects that
   grammar or not. You will first need to write a function capable of generating a training batch
   containing about 50% strings that respect the grammar, and 50% that don’t.**

    _See notebooks_

9. **Train an Encoder–Decoder model that can convert a date string from one format to another (e.g.,
   from “April 22, 2019” to “2019-04-22”).**

    _See notebooks_

10. **Go through TensorFlow’s Neural Machine Translation with Attention tutorial.**

    _See notebooks_

11. **Use one of the recent language models (e.g., BERT) to generate more convincing Shakespearean
    text.**

    _See notebooks_

