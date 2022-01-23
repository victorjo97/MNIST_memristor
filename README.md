# MNIST_memristor
Python code implementing the algorithms of my MSc Dissertation on "Modified training for increased robustness in memristor-based neural networks".

The purpose of the project was to investigate a hardware accelerator for Deep Learning based on the use of a novel electronic device called memristor. 
These devices arranged in crossbar arrays directly implement the conceptual structure of artificial neural networks. Due to this similarity,
memristor-based artificial neural networks are expected to perform more rapidly and with a lower energetic cost that the present implementetions
in digital general-purpose computing devices.

However, as in any hardware implementation, there are non-idealities which distort the implemented deep learning models from the ideal ones. The Dissertation above mentioned investigated how the training algorithm can be tuned to minimise the damages on the models' performance when using memristor crossbar arrays.

MNIST.py was used as an introduction to neural networks by coding with numpy the algorithm that trains neural networks for the MNIST classification problem of handwritten digits.

torch_MNSIT.py implements the training of neural networks facing the MNIST classification problem using the tools of PyTorch.

data_handling.py creates the functions used to extract and update the synaptic parameters of the models created via torch_MNIST.py.

non_idealities.py implements the non-idealities considered in this work and the measure of its effects. 

You can find the associated Msc Dissertation here: https://sites.google.com/view/vzamora/assignments
