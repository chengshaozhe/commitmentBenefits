import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pickle
import random
import json
from collections import OrderedDict
import time
import src.MDPChasing.envDiscreteGrid as env
from src.MDPChasing.envDiscreteGrid import IsTerminal, Transition, Reset
from src.MDPChasing.reward import RewardFunctionCompete, RewardFunction
from src.MDPChasing.state import GetAgentPosFromState

from src.MDPChasing.policies import stationaryAgentPolicy, RandomPolicy
from src.chooseFromDistribution import maxFromDistribution
from src.visualization import *
from src.trajectory import SampleTrajectory



def main():
    lowerBound = 0
    gridSize = 10
    upperBound = [gridSize - 1, gridSize - 1]

    actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    numActionSpace = len(actionSpace)

    sheepSpeedRatio = 1
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * sheepSpeedRatio))

    numOfAgent = 5  # 1 hunter, 1 stag with high value, 3 rabbits with low value
    hunterId = [0]
    targetIds = [1, 2, 3, 4]
    stagId = [1]
    rabbitId = [2, 3, 4]

    positionIndex = [0, 1]

    getHunterPos = GetAgentPosFromState(hunterId, positionIndex)
    getTargetsPos = GetAgentPosFromState(targetIds, positionIndex)

    stayWithinBoundary = env.StayWithinBoundary(upperBound, lowerBound)
    isTerminal = env.IsTerminal(getHunterPos, getTargetsPos)
    transitionFunction = env.Transition(stayWithinBoundary)
    reset = env.Reset(upperBound, lowerBound, numOfAgent)

    stagPolicy = RandomPolicy(sheepActionSpace)
    stagPolicy = stationaryAgentPolicy

    rabbitPolicies = [stationaryAgentPolicy] * len(rabbitId)







if __name__ == '__main__':
    main()