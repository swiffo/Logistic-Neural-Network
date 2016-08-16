"""Unit tests for the logistic neural network program.

    Some fairly simple testing using the 'unittest' framework. 
"""

import unittest
import numpy as np
import math
import neural_network as neural


class TestSigmoid(unittest.TestCase):
    """Test for correctness of the sigmoid function, sigmoid.

    The sigmoid function is x -> 1/(1+e^{-x}).
    """
    
    def test_sigmoid(self):
        """Test sigmoid function for specific cases."""
        m = np.array([[3, 4], [-1, -2], [-3.5, 4.5]])
        m_c = m.copy()

        output = neural.sigmoid(m)
        self.assertEqual(output.shape, m_c.shape)

        for row in range(m_c.shape[0]):
            for col in range(m_c.shape[1]):
                self.assertEqual(output[row,col], 1/(1+math.exp(-m_c[row, col])))

        self.assertTrue((m==m_c).all())

class TestSigmoidPrime(unittest.TestCase):
    """Test for correctness of the sigmoid_prime function.

    sigmoid_prime should give the derivative of the sigmoid function,
    x -> 1/(1+e^{-x}). That is, x -> sigma(x) * (1-sigma(x)) where
    sigma is the sigmoid function.
    """
    
    def test_sigmoid_prime(self):
        """Test the sigmoid_prime function for specific inputs."""
        m = np.array([[1,2], [-3,4]])
        m_c = m.copy()
        eps = 1e-6

        der = neural.sigmoid_prime(m)
        exp = (neural.sigmoid(m+eps) - neural.sigmoid(m-eps)) / ( 2*eps)

        diff = np.sum(der - exp)
        self.assertAlmostEqual(diff, 0, delta=1e-6)

        m_unchanged = (m==m_c).all()
        self.assertTrue(m_unchanged)    

class Testsignal_to_activation(unittest.TestCase):
    """Test for the neuron activation signal.

    Test that the input signal to a neuron is correctly converted to
    activation along and that the bias unit is present.
    """
    
    def test_signal_to_activation(self):
        """Test the activation from specific signal."""
        m = np.array([[1,2],[-3,4]])
        m_c = m.copy()

        act = neural.signal_to_activation(m)
        shape_ok = act.shape[0] == m.shape[0]+1 and act.shape[1] == m.shape[1]
        bias_ok = (act[0,:] == np.ones((1,m.shape[1]))).all()
        nonbias_ok = (act[1:,:] == neural.sigmoid(m)).all()

        self.assertTrue(shape_ok)
        self.assertTrue(bias_ok)
        self.assertTrue(nonbias_ok)
        self.assertTrue((m==m_c).all())
       

if __name__=='__main__':
    unittest.main()
