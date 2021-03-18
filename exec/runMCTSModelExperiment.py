import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import collections as co
import itertools as it
from collections import OrderedDict

from src.algorithms.mcts import MCTS, ScoreChild, establishSoftmaxActionDist,establishPlainActionDist, SelectChild, Expand, RollOut, backup, InitializeChildren
import src.MDPChasing.envDiscreteGrid as env
from src.controller import ModelControllerMCTS
from src.MDPChasing.state import GetAgentPosFromState
import src.MDPChasing.reward as reward
from src.MDPChasing.policies import stationaryAgentPolicy, RandomPolicy
from src.chooseFromDistribution import maxFromDistribution
from src.visualization import *
from src.writer import WriteDataFrameToCSV, loadFromPickle

from src.simulationTrial import NormalTrialMCTSMaze



def mctsPolicy(condition):
    renderOn = True
    numSimulations = condition['numSimulations']
    maxRolloutSteps = condition['maxRolloutSteps']

    resultsPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'results'))

    gridSize = 15
    lowerBound = 0
    upperBound = [gridSize - 1, gridSize - 1]

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
    targetColor = [221, 160, 221]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    textColorTuple = (255, 50, 50)

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth,
                                    textColorTuple)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    drawText = DrawText(screen, drawBackground)



    actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    # actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1),(1, -1), (1, 1), (-1, 1), (-1, -1)]
    numActionSpace = len(actionSpace)


    numOfAgent = 5  # 1 hunter, 1 stag with high value, 3 rabbits with low value
    hunterId = [0]
    stagId = [1]
    rabbitId = [2, 3, 4]
    # targetIds = stagId + rabbitId
    targetIds = stagId

    positionIndex = [0, 1]
    getHunterPos_ = GetAgentPosFromState(hunterId, positionIndex)
    getHunterPos = lambda state: getHunterPos_(state[0])
    getTargetsPos_ = GetAgentPosFromState(targetIds, positionIndex)
    getTargetsPos = lambda state: getTargetsPos_(state[0])

    getStaqPos_ = GetAgentPosFromState(stagId, positionIndex)
    getStaqPos = lambda state: getStaqPos_(state[0])

    stayWithinBoundary = env.StayWithinBoundaryMaze(upperBound, lowerBound)
    # isTerminal = env.IsTerminal(getHunterPos, getTargetsPos)
    killZone=0
    isTerminal = env.IsTerminalWithKillZone(getHunterPos, getTargetsPos,killZone)


    highValueSteps = 19
    numberOfMaze = 5
    loadPath = os.path.join('..', 'pathPlanning', 'map', str(highValueSteps) + 'highValueSteps' + str(numberOfMaze) + 'maps' + '.pickle')
    mazeList = loadFromPickle(loadPath)
    reset = env.ResetMaze(upperBound, lowerBound, mazeList)

    transitionFunction = env.TransitionWithObstacles(stayWithinBoundary)


    # stagPolicy = RandomPolicy(sheepActionSpace)
    stagPolicy = stationaryAgentPolicy
    rabbitPolicies = [stationaryAgentPolicy] * len(rabbitId)


    #MCTS
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)
    getActionPrior = lambda state: {action: 1 / len(actionSpace) for action in actionSpace}

    # def wolfTransit(state, action): return transitionFunction(
        # state, [action, maxFromDistribution(stagPolicy(state))] + [maxFromDistribution(rabbitPolicy(state)) for rabbitPolicy in rabbitPolicies])
    def wolfTransit(state, action): return transitionFunction(
        state, [action, maxFromDistribution(stagPolicy(state))])

    maxRunningSteps = 10
    stepPenalty = -1 / maxRunningSteps
    # stepPenalty = -0
    # .1
    catchBonus = 1
    highRewardRatio = 2
    rewardFunction = reward.RewardFunction(highRewardRatio, stepPenalty, catchBonus, isTerminal,getHunterPos,getStaqPos)

    initializeChildren = InitializeChildren(actionSpace, wolfTransit, getActionPrior)
    expand = Expand(isTerminal, initializeChildren)
    def rolloutPolicy(state): return actionSpace[np.random.choice(range(numActionSpace))]
    rolloutHeuristicWeight = 0
    rolloutHeuristic = reward.HeuristicDistanceToTarget(rolloutHeuristicWeight, getHunterPos, getTargetsPos)
    # rolloutHeuristic = lambda state: 0

    rollout = RollOut(rolloutPolicy, maxRolloutSteps, wolfTransit,
                      rewardFunction, isTerminal, rolloutHeuristic)
    wolfPolicy = MCTS(numSimulations, selectChild, expand,
                      rollout, backup, establishPlainActionDist)

    # All agents' policies
    policy = lambda state: [wolfPolicy(state), stagPolicy(state)] + [rabbitPolicy(state) for rabbitPolicy in rabbitPolicies]

    softmaxBeta=-1
    for i in range(1):
        print(i)
        # expDesignValues = [[condition, diff] for condition in conditionList for diff in targetDiffsList] * numBlocks
        # random.shuffle(expDesignValues)
        # expDesignValues.append(specialDesign)

        modelController = ModelControllerMCTS(softmaxBeta)
        # modelController = AvoidCommitModel(softmaxBeta, actionSpace, checkBoundary)
        controller = modelController


        normalTrial = NormalTrialMCTSMaze(renderOn, controller, drawNewState, drawText,transitionFunction,isTerminal)

        experimentValues = co.OrderedDict()
        experimentValues["name"] = "maxRolloutSteps" + str(maxRolloutSteps) + '_' + "numSimulations" + str(numSimulations) + '_' + "softmaxBeta" + str(softmaxBeta) + '_' + str(i)
        resultsDirPath = os.path.join(resultsPath,  "softmaxBeta" + str(softmaxBeta))
        if not os.path.exists(resultsDirPath):
            os.mkdir(resultsDirPath)

        writerPath = os.path.join(resultsDirPath, experimentValues["name"] + '.csv')
        writer = WriteDataFrameToCSV(writerPath)
        experiment = ModelSimulationMazeMCTS(reset, normalTrial, writer, experimentValues, resultsPath)
        conditionList=range(100)
        experiment(conditionList,policy)

class ModelSimulationMazeMCTS():
    def __init__(self, creatMap, normalTrial, writer, experimentValues, resultsPath):
        self.creatMap = creatMap
        self.normalTrial = normalTrial
        self.writer = writer
        self.experimentValues = experimentValues
        self.resultsPath = resultsPath

    def __call__(self, conditionList,policy):
        for trialIndex, condition in enumerate(conditionList):
            initialState = self.creatMap()
            results = self.normalTrial(initialState, policy)


            # results["conditionName"] = condition.name
            # results["decisionSteps"] = str(condition.decisionSteps)
            # results["obstacles"] = str(obstacles)



            response = self.experimentValues.copy()
            response.update(results)
            self.writer(response, trialIndex)

def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numSimulations'] = [200, 400, 600]#[0.0, 1.0]
    manipulatedVariables['maxRolloutSteps'] =[20, 30]# [0.0, 0.2, 0.4]
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditions = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    for condition in conditions:
        print(condition)
        mctsPolicy(condition)
        # try:
        #     mctsPolicy(condition)
        # except:
        #     continue


if __name__ == "__main__":
    main()

