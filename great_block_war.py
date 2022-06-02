import random

import grpc
import numpy as np
from numpy import atleast_2d, tanh, random, array, dot, ones

import minecraft_pb2_grpc
from minecraft_pb2 import *

blocksUsedList = [
    'EMERALD_BLOCK',
    'DIAMOND_BLOCK'
]

neighbor = array([2, 2])

channel = grpc.insecure_channel('localhost:5001')
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

population = []

# <editor-fold desc="INIT BASEMENT">

client.fillCube(FillCubeRequest(  # Clear a 40x100x40 working area
    cube=Cube(
        min=Point(x=-100, y=0, z=-100),
        max=Point(x=100, y=100, z=100)
    ),
    type=AIR
))

client.fillCube(FillCubeRequest(  # Create a 40x10x40 working area
    cube=Cube(
        min=Point(x=0, y=4, z=0),
        max=Point(x=20, y=4, z=20)
    ),
    type=QUARTZ_BLOCK
))


# </editor-fold>

# <editor-fold desc="SPAWNING BLOCKS">

def spawnRandomBlock(x, y, z, block):
    client.spawnBlocks(Blocks(blocks=[
        Block(position=Point(x=x, y=y, z=z), type=block, orientation=NORTH)
    ]))
    pass


# </editor-fold>

# <editor-fold desc="GENERATE POPULATION">

for i in range(20):
    for j in range(20):
        block = random.choice(blocksUsedList)

        if random.choice([0, 1]) == 1:
            spawnRandomBlock(i, 5, j, block)


# </editor-fold>

# <editor-fold desc="MLP ALGORITHM">

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return tanh(x)


# derivative of our sigmoid function
def dsigmoid(x):
    return 1.0 - x ** 2


def batlleWithOtherKnight(intention, x, z):
    if intention == -0.5:
        spawnRandomBlock(x, 5, z, "REDSTONE_BLOCK")
        print("Emerald has died...")

    if intention == 0.5:
        spawnRandomBlock(x, 5, z, "GLOWSTONE")
        print("Diamond won !!")


def roundToInteger(pattern):
    return np.round(pattern * 2) / 2


def getNeighbor(x: int, z: int):
    currentBlock = client.readCube(Cube(min=Point(x=x, y=5, z=z), max=Point(x=x, y=5, z=z)))
    currentNeighbor = client.readCube(Cube(min=Point(x=x + 1, y=5, z=z), max=Point(x=x + 1, y=5, z=z)))

    if currentBlock.blocks[0].type == 5 or currentNeighbor.blocks[0].type == 5:
        neighbor[0] = 2
        neighbor[1] = 2
        return

    if currentBlock.blocks[0].type == 68:
        neighbor[0] = 0
    if currentNeighbor.blocks[0].type == 68:
        neighbor[1] = 0

    if currentBlock.blocks[0].type == 58:
        neighbor[0] = 1
    if currentNeighbor.blocks[0].type == 58:
        neighbor[1] = 1


class MLP:
    def __init__(self, *args):
        self.args = args
        n = len(args)

        self.layers = [ones(args[i] + (i == 0)) for i in range(0, n)]

        self.weights = list()
        for i in range(n - 1):
            R = random.random((self.layers[i].size, self.layers[i + 1].size))
            self.weights.append((2 * R - 1) * 0.20)

        self.m = [0 for i in range(len(self.weights))]

    def update(self, inputs):
        self.layers[0][:-1] = inputs

        for i in range(1, len(self.layers)):
            self.layers[i] = sigmoid(dot(self.layers[i - 1], self.weights[i - 1]))

        return self.layers[-1]

    def backPropagate(self, inputs, outputs, a=0.1, m=0.1):

        error = outputs - self.update(inputs)
        de = error * dsigmoid(self.layers[-1])
        deltas = list()
        deltas.append(de)

        for i in range(len(self.layers) - 2, 0, -1):
            deh = dot(deltas[-1], self.weights[i].T) * dsigmoid(self.layers[i])
            deltas.append(deh)

        deltas.reverse()

        for i, j in enumerate(self.weights):
            layer = atleast_2d(self.layers[i])
            delta = atleast_2d(deltas[i])

            dw = dot(layer.T, delta)
            self.weights[i] += a * dw + m * self.m[i]
            self.m[i] = dw


pat = (((0, 0), 0),
       ((0, 1), -0.5),
       ((1, 0), 0.5),
       ((1, 1), 0),
       ((2, 2), 1))

n = MLP(2, 2, 1, 1)

for i in range(1000):
    for p in pat:
        n.backPropagate(p[0], p[1])

for i in range(20):
    for j in range(20):
        for p in pat:
            getNeighbor(i, j)
            checkPattern = n.update(neighbor)
            neighbor[0] = 2
            neighbor[1] = 2
            val = roundToInteger(checkPattern)
            batlleWithOtherKnight(val, i, j)


# </editor-fold>
