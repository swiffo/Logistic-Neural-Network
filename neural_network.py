import numpy as np
import math
import random

def sigmoid(m):
    """Sigmoid function vectorized"""
    return 1/(1+np.exp(-m))

def sigmoidPrime(m):
    """Sigmoid derivative vectorized"""
    return sigmoid(m)*(1-sigmoid(m))

def signalToActivation(z):
    """z is matrix of size numLayers x numTrainingExamples.
Output is matrix of size (numLayers+1) x numTrainingExamples"""
    numLayers, numTrainingExamples = z.shape
    activation = np.vstack( (np.ones(numTrainingExamples), sigmoid(z)) )
    return activation

class MatrixDimensionException(BaseException):
    def __init__(self, text):
        BaseException.__init__(self, text)
        
class NN():
    """Neural network. Initialized with number of neurons in each layer (excluding bias neuron)"""
    
    def __init__(self, layers):
        self._layers = layers # Array of depths, not including bias

    def setTrainingData(self, inputs, outputs):
        """Rows of inputs and inputs must match (each row being one observation)
inputs and outputs must be 2d ndarray
"""

        input_rows, input_depth = inputs.shape
        output_rows, output_depth = outputs.shape

        if input_rows != output_rows:
            raise MatrixDimensionException("Different number of inputs and ouputs. {} vs {}.".format(input_rows, output_rows))

        self._N = input_rows # Number of training observations
        
        if input_depth != self._layers[0]:
            raise MatrixDimensionException("Wrong number of input signals")

        if output_depth != self._layers[-1]:
            raise MatrixDimensionException("Wrong number of output signals")

        # inputs are stored as signals (stored as depth x numTrainingObservations)
        self._outputs=outputs.T

        # Set the signals (empty to begin with). Does not include bias term (which would be infinite)
        self._signals = [np.empty((depth, self._N)) for depth in self._layers]
        self._signals[0] = np.transpose(inputs)

        # Set the activations (includes bias term)
        self._activations = [np.empty((depth+1, self._N)) for depth in self._layers]
        act0 = np.vstack((np.ones(self._N), sigmoid(self._signals[0])))
        self._activations[0] = act0

        # Set the transition matrices
        self.initThetas()

        # set the epsilon matrices 
        self._epsilons = [np.empty((depth, self._N)) for depth in self._layers]

    def initThetas(self):
        """Randomly initialize the transition matrices"""
        from_layers = self._layers[:-1]
        to_layers = self._layers[1:]
        self._thetas = [np.random.random((t, f+1)) for (f, t) in zip(from_layers, to_layers)]

    def feedForward(self):
        """Feeds input signal through the network, populating all signals and activations"""
        for (from_layer, theta) in enumerate(self._thetas):
            to_layer = from_layer+1
            to_signal = np.dot(theta, self._activations[from_layer])
            to_activation = signalToActivation(to_signal) # Adds bias neuron

            self._signals[to_layer]=to_signal
            self._activations[to_layer]=to_activation

    def _lastLayerActivation(self):
        """Returns matrix of size depth x numTrainingObservations of activations.
Note that there is no bias term!"""
        
        activations=self._activations[-1]
        activations=activations[1:,:] # Remove bias term (does not make sense for last layer)
        
        return activations
            
    def totalError(self):
        """The total logistic error of the network"""
        activations=self._lastLayerActivation()
        outputs = self._outputs

        errorMatrix= (-1/self._N) * ( outputs*np.log(activations) + (1-outputs)*np.log(1-activations) )
        error = np.sum(errorMatrix)
        
        return error

    def _errorDerivMatrix(self):
        """Matrix of size depth x numTrainingObservations.
Derivative of total error with respect to signal (not activation) of last layer of neurons per observation."""
        
        activations = self._lastLayerActivation() # no bias neuron
        outputs = self._outputs
        errorDer = (-1/self._N) * ( outputs*(1-activations) - (1-outputs)*activations )

        return errorDer

    def _populateEpsilons(self):
        """Populates the epsilon-matrices (size depth x numTrainingObservations).
These are the error term differentiated by signal"""
        
        self._epsilons[-1] = self._errorDerivMatrix()
        for index in range(len(self._layers)-2,0,-1): # deliberately omitting index=0 
            postEps = self._epsilons[index+1]
            theta = self._thetas[index]
            act = self._activations[index][1:,:] # Remove bias term

            eps = np.dot(theta.T, postEps)[1:,:] * act * (1-act)
            self._epsilons[index]=eps
        
    def _backPropagate(self, alpha):
        """Update all thetas along gradient with speed minus alpha"""
        self._populateEpsilons()

        newThetas = []
        for index, theta in enumerate(self._thetas):
            act = self._activations[index]
            postEps = self._epsilons[index+1]
            thetaDer = np.dot(postEps, act.T)

            newThetas.append(theta - alpha * thetaDer)

        self._thetas = newThetas

    def train(self, iterations, verbose=False):
        """Run [iterations] of feedforwards and back-propagations"""
        for i in range(iterations):
            self.feedForward()
            self._backPropagate(0.5)

            if verbose:
                error = self.totalError()
                print("It {} error = {:.2f}".format(i, error))

    def predict(self, inputs):
        """Predict using current state of neural network"""
        # Set the signals (empty to begin with). Does not include bias term (which would be infinite)
        self._signals[0] = np.transpose(inputs)
        self._N = inputs.shape[0]

        # Set the activations (includes bias term)
        act0 = np.vstack((np.ones(self._N), sigmoid(self._signals[0])))
        self._activations[0] = act0

        self.feedForward()

        return self._lastLayerActivation()
