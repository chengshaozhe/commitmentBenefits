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
from src.MDPChasing.envDiscreteGrid import IsTerminal, TransiteForNoPhysics, Reset

from src.MDPChasing.state import GetAgentPosFromState
import src.MDPChasing.reward as reward
from src.MDPChasing.policies import stationaryAgentPolicy
from src.trajectory import SampleTrajectory
from src.chooseFromDistribution import maxFromDistribution


def main():
    lowerBoundary, gridSize = [0, 15]

    numSimulations = 50
    maxRolloutSteps = 10

    actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    numActionSpace = len(actionSpace)

    sheepSpeedRatio = 1
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * sheepSpeedRatio))

    numOfAgent = 4
    wolfId = 0
    sheepIds = [1, 2, 3]
    sheepOneId = 1
    sheepTwoId = 2
    sheepThreeId = 3

    positionIndex = [0, 1]

    getWolfPos = GetAgentPosFromState(wolfId, positionIndex)
    getSheepPos = GetAgentPosFromState(sheepIds, positionIndex)

    stayWithinBoundary = env.StayWithinBoundary(gridSize, lowerBoundary)

    isTerminal = env.IsTerminal(getWolfPos, getSheepPos)

    transitionFunction = env.Transition(stayWithinBoundary)
    reset = env.Reset(xBoundary, yBoundary, numOfAgent)

    sheepOnePolicy = RandomPolicy(sheepActionSpace)
    sheepTwoPolicy = stationaryAgentPolicy
    sheepThreePolicy = stationaryAgentPolicy

    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)
    getActionPrior = lambda state: {action: 1 / len(actionSpace) for action in actionSpace}

    def wolfTransit(state, action): return transitionFunction(
        state, [action, chooseGreedyAction(sheepOnePolicy(state))])

    maxRunningSteps = 100
    stepPenalty = -1 / maxRunningSteps
    catchBonus = 1
    rewardFunction = reward.RewardFunctionCompete(
        stepPenalty, catchBonus, isTerminal)

    initializeChildren = InitializeChildren(
        actionSpace, wolfTransit, getActionPrior)
    expand = Expand(isTerminal, initializeChildren)

    def rolloutPolicy(
        state): return actionSpace[np.random.choice(range(numActionSpace))]
    rolloutHeuristicWeight = 1
    rolloutHeuristic = reward.HeuristicDistanceToTarget(
        rolloutHeuristicWeight, getPredatorOnePos, getPreyPos)

    rollout = RollOut(rolloutPolicy, maxRolloutSteps, wolfTransit,
                      rewardFunction, isTerminal, rolloutHeuristic)

    wolfPolicy = MCTS(numSimulations, selectChild, expand,
                      rollout, backup, establishSoftmaxActionDist)

    # All agents' policies
    policy = lambda state: [wolfPolicy(state), sheepOnePolicy(state), sheepTwoPolicy(state), sheepThreePolicy(state)]

    chooseAction = [maxFromDistribution] * 4
    sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transitionFunction, isTerminal, reset, chooseAction)

    startTime = time.time()
    trajectories = [sampleTrajectory(policy)]

    finshedTime = time.time() - startTime

    print('lenght:', len(trajectories[0]))

    print('time:', finshedTime)


if __name__ == "__main__":
    main()
