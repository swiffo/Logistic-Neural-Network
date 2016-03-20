import neural_network as neural
import numpy as np
import random

def greaterThan():
    """Should discover whether numbers between 0 and 10 are greater than 5"""

    inputs=np.random.random((1000, 1))*10
    outputs = (inputs > 5 ) * 1

    nn=neural.NN([1,2,1])
    nn.setTrainingData(inputs, outputs)
    nn.train(10000)

    test_inputs = np.linspace(0,10,15).reshape((15,1))
    test_outputs = nn.predict(test_inputs)

    collected = np.hstack((test_inputs, test_outputs.T))
    for row in collected:
        i = row[0]
        o = row[1]
        print("{:.1f} > 5? {}  ({:.2f})".format(i, o > 0.5, o))


def onAndOff():
    """Maps single 'off' neuron to 1 and 0, and 'on' to 0 and 1"""    

    data = np.array([
        [-100, 1, 0 ],
        [100, 0, 1 ],
    ] )

    inputs = data[:,[0]]
    outputs = data[:,1:]

    nn=neural.NN([1,3,3,2])
    nn.setTrainingData(inputs, outputs)
    nn.train(5000)
    print(nn.predict(inputs))


def isTriangle(x,y,z):
    """Checks whether a triangle exists with the side lengths x,y,z"""
    
    lengths = [x,y,z]
    
    if sum([x <= 0 for x in lengths]) > 0:
        raise BaseException() # Still being lazy

    lengths.sort()

    return(lengths[0] + lengths[1] > lengths[2])


def triangleNetwork():
    """Builds neural network to check whether 3 sticks of specified length can form a triangle"""

    shortest = 1
    longest  = 100
    rows = []
    for _n in range(25000):
        x = random.randint(shortest, longest)
        y = random.randint(shortest, longest)
        z = random.randint(shortest, longest)
        rows.append([x,y,z, float(isTriangle(x,y,z))])

    data = np.array(rows)
    inputs = data[:,0:3]
    outputs = data[:,[3]]

    nn = neural.NN([3,5,5,1])
    nn.setTrainingData(inputs, outputs)
    nn.train(10000)

    rows = []
    for _n in range(50):
        x = random.randint(shortest, longest)
        y = random.randint(shortest, longest)
        z = random.randint(shortest, longest)
        rows.append([x,y,z])

    test_inputs = np.array(rows)
    test_results = nn.predict(test_inputs)

    totalCount=0
    correctCount=0
    for xx,res in zip(test_inputs, test_results[0]):
        isCorrect = isTriangle(xx[0],xx[1],xx[2]) == (res > 0.5)
        print("({:2d},{:2d},{:2d}) :: {:^5} ({:.2f}).".format(xx[0], xx[1], xx[2], str(isCorrect), res))
        totalCount += 1
        correctCount += float(isCorrect)
    
