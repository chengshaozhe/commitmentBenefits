import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pickle
import random
import json
from collections import OrderedDict
import time

from src.algorithms.mcts import MCTS, ScoreChild, establishSoftmaxActionDist, SelectChild, Expand, RollOut, backup, InitializeChildren
import src.MDPChasing.envDiscreteGrid as env
from src.MDPChasing.envDiscreteGrid import IsTerminal, Transition, Reset

from src.MDPChasing.state import GetAgentPosFromState
import src.MDPChasing.reward as reward
from src.MDPChasing.policies import stationaryAgentPolicy, RandomPolicy
from src.trajectory import SampleTrajectory
from src.chooseFromDistribution import maxFromDistribution
from src.visualization import *




def mctsPolicy():
    lowerBound = 0
    gridSize = 10
    upperBound = [gridSize - 1, gridSize - 1]

    maxRunningSteps = 50

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

    # stagPolicy = RandomPolicy(sheepActionSpace)
    stagPolicy = stationaryAgentPolicy
    rabbitPolicies = [stationaryAgentPolicy] * len(rabbitId)

    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)
    getActionPrior = lambda state: {action: 1 / len(actionSpace) for action in actionSpace}

    def wolfTransit(state, action): return transitionFunction(
        state, [action, maxFromDistribution(stagPolicy(state))] + [maxFromDistribution(rabbitPolicy(state)) for rabbitPolicy in rabbitPolicies])

    stepPenalty = -1 / maxRunningSteps
    catchBonus = 1
    highRewardRatio = 10
    rewardFunction = reward.RewardFunction(highRewardRatio, stepPenalty, catchBonus, isTerminal)

    initializeChildren = InitializeChildren(actionSpace, wolfTransit, getActionPrior)
    expand = Expand(isTerminal, initializeChildren)

    def rolloutPolicy(state): return actionSpace[np.random.choice(range(numActionSpace))]
    rolloutHeuristicWeight = 0
    rolloutHeuristic = reward.HeuristicDistanceToTarget(rolloutHeuristicWeight, getHunterPos, getTargetsPos)
    # rolloutHeuristic = lambda state: 0

    maxRolloutSteps = 20
    rollout = RollOut(rolloutPolicy, maxRolloutSteps, wolfTransit,
                      rewardFunction, isTerminal, rolloutHeuristic)
    numSimulations = 600
    wolfPolicy = MCTS(numSimulations, selectChild, expand,
                      rollout, backup, establishSoftmaxActionDist)

    # All agents' policies
    policy = lambda state: [wolfPolicy(state), stagPolicy(state)] + [rabbitPolicy(state) for rabbitPolicy in rabbitPolicies]
    numOfAgent = 5
    gridSize = 15
    maxRunningSteps = 50

    screenWidth = 600
    screenHeight = 600
    fullScreen = False

    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()
    # pg.mouse.set_visible(False)

    leaveEdgeSpace = 2
    lineWidth = 1
    backgroundColor = [205, 255, 204]
    lineColor = [0, 0, 0]
    distractorColor = [255,255,0]
    targetColor = [221,160,221]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    textColorTuple = (255, 50, 50)

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)

    chooseAction = [maxFromDistribution] * numOfAgent

    renderOn = True
    sampleTrajectory = SampleTrajectory(maxRunningSteps, transitionFunction, isTerminal, reset, chooseAction, renderOn, drawNewState)

    startTime = time.time()
    numOfEpisodes = 10
    trajectories = [sampleTrajectory(policy) for i in range(numOfEpisodes)]
    finshedTime = time.time() - startTime

    print('lenght:', len(trajectories[0]))
    print('time:', finshedTime)


if __name__ == "__main__":
    mctsPolicy()

