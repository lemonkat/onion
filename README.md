# onion

Onion, named because it contains layers, is a custom neural network library I wrote as an extra project for a high school machine learning class.

Onion does not have GPU acceleration, being built straight on top of NumPy. It does have automatic differentiation, batched training, and a variety of common layer types.

Making Onion was an exercise in wrangling NumPy for maximal performance, but the main challenge was how to design each component to fit together well in a way that would be convenient to use for the end user. Making new layer types had to be straightforward and easy to debug.

Onion's architecture is built around the Layer, a callable object which retains its most recent parameters for gradient tracking. A Layer object will recieve its inputs during the forward pass, store them and the calculated outputs, and then use these to compute its input gradients from its output gradients during the backward pass. This is similar to how PyTorch and TensorFlow work. 

Layers are assembled into Models, which connect the inputs and outputs of several Layers and loss functions. A topological sorting algorithm determines which layers need to be run in which order to produce a desired output, whether that is the network's output or the loss value, and then run the backpropagation pass.

Other features Onion has:
 - model saving and loading via .onn file format
 - regularization and support for multiple loss functions
 - an (untested) wrapper for PyTorch model objects
 - utilities for training and plotting metrics on separate processes, as well as hyperparameter tuning
 - batch norm, layer norm, convolution, and pooling layers (No RNNs or transformers b/c my class switched to PyTorch by time we learned about those, but they could be added)

Speaking of which, at some point I may revisit this project. Here's some things I want to add:
 - a more generalized computational graph system to see which operations may be computed at the same time, enabling parallelization
 - integration with Tensorboard
 - GPU support? (won't be easy and I honestly have no idea how to do this yet)
 - more layer types (more optimizers, more loss functions, RNNs, attention, etc)
 - better documentation and testing, up to the same standard as my latest projects
