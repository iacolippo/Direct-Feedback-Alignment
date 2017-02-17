# Direct-Feedback-Alignment

In *dfa-linear-net.ipynb*, I show how a neural network without activation function can learn a linear function (multiplication by a matrix) using direct feedback alignment (DFA), as in [Nøkland, 2016](https://arxiv.org/pdf/1609.01596.pdf). There is also some theory about it.

In *dfa-mnist.ipynb*, I show how a neural network trained with DFA achieves very similar results to one trained with backpropagation. The architecture is very simple: one hidden layer of 800 Tanh units, sigmoid in the last layer and binary crossentropy loss.
