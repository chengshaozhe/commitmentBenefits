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
from src.algorithms.qLearning import GetAction, UpdateQTable, initQtable, argMax, QLearningAgent
from src.chooseFromDistribution import maxFromDistribution
from src.visualization import *
from src.trajectory import SampleTrajectory


def calculateSoftmaxProbability(acionValues, beta):
    newProbabilityList = list(np.divide(np.exp(np.multiply(beta, acionValues)), np.sum(np.exp(np.multiply(beta, acionValues)))))

    return newProbabilityList


class SoftmaxPolicy:
    def __init__(self, softmaxBeta, QDict, actionSpace):
        self.QDict = QDict
        self.softmaxBeta = softmaxBeta
        self.actionSpace = actionSpace

    def __call__(self, state):
        actionValues = self.QDict[str(state)]
        softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
        softMaxActionDict = dict(zip(self.actionSpace, softmaxProbabilityList))
        return softMaxActionDict


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
    getStaqPos = GetAgentPosFromState(stagId, positionIndex)
    getTargetsPos = GetAgentPosFromState(targetIds, positionIndex)

    stayWithinBoundary = env.StayWithinBoundary(upperBound, lowerBound)
    isTerminal = env.IsTerminal(getHunterPos, getTargetsPos)
    transitionFunction = env.Transition(stayWithinBoundary)
    reset = env.Reset(upperBound, lowerBound, numOfAgent)
    state = reset()
    fixReset = lambda :state

    fixReset = lambda: [(1, 1), (8, 8), (8, 9), (9, 9), (9, 8)]
    stagPolicy = RandomPolicy(sheepActionSpace)
    stagPolicy = stationaryAgentPolicy

    rabbitPolicies = [stationaryAgentPolicy] * len(rabbitId)

    def wolfTransit(state, action): return transitionFunction(
        state, [action, maxFromDistribution(stagPolicy(state))] + [maxFromDistribution(rabbitPolicy(state)) for rabbitPolicy in rabbitPolicies])

    maxRunningSteps = 100
    stepPenalty = -1 / maxRunningSteps
    catchBonus = 1
    highRewardRatio = 1
    rewardFunction = RewardFunction(highRewardRatio, stepPenalty, catchBonus, isTerminal,getHunterPos,getStaqPos)

# q-learing
    initQ = initQtable(actionSpace)
    discountFactor = 0.9
    learningRate = 0.01
    updateQTable = UpdateQTable(discountFactor, learningRate)
    epsilon = 0.1
    getAction = GetAction(epsilon, actionSpace, argMax)

    episodes = 2000
    maxRunningStepsPerEpisodes = 100
    qLearningAgent=QLearningAgent(initQ, episodes, maxRunningStepsPerEpisodes, fixReset, wolfTransit, isTerminal, rewardFunction, updateQTable, getAction)
    startTime = time.time()
    QDict = qLearningAgent()
    finshedTime = time.time() - startTime
    print('time:', finshedTime)
    # startTime = time.time()
    # initstate = reset()
    # for episode in range(2):
    #     state = initstate
    #     # state = reset()
    #     for step in range(maxRunningSteps):
    #         wolfactionIndex = getAction(qTable, str(state))
    #         wolfaction = actionSpace[wolfactionIndex]
    #         nextState = wolfTransit(state, wolfaction)
    #         reward = rewardFunction(nextState, wolfaction)
    #         done = isTerminal(nextState)
    #
    #         qTable = updateQTable(qTable, str(state), wolfactionIndex, reward, str(nextState))
    #         state = nextState
    #         if done:
    #
    #             break

    # QDict = qTable
    print(QDict)
    softmaxBeta = 5
    wolfPolicy = SoftmaxPolicy(softmaxBeta, QDict, actionSpace)
    # All agents' policies
    policy = lambda state: [wolfPolicy(state), stagPolicy(state)] + [rabbitPolicy(state) for rabbitPolicy in rabbitPolicies]

    screenWidth = 600
    screenHeight = 600
    fullScreen = False
    numOfAgent = 5
    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()
    # pg.mouse.set_visible(False)



    maxRunningSteps = 100
    leaveEdgeSpace = 2
    lineWidth = 1
    backgroundColor = [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    textColorTuple = (255, 50, 50)

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)

    chooseAction = [maxFromDistribution] * numOfAgent

    renderOn = True
    # print()
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transitionFunction, isTerminal, fixReset, chooseAction, renderOn, drawNewState)


    numOfEpisodes = 3000
    trajectories = [sampleTrajectory(policy) for i in range(numOfEpisodes)]

    print('lenght:', len(trajectories[0]))

if __name__ == "__main__":
    main()
