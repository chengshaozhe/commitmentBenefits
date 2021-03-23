import os
import sys
os.chdir(sys.path[0])
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pickle
import random
import json
import time

from collections import OrderedDict
import collections as co
import itertools as it
import numpy as np

import src.MDPChasing.envDiscreteGrid as env
from src.MDPChasing.envDiscreteGrid import IsTerminal, Transition, ResetMazeNoWallForTrain,ResetMazeNoWallFix
from src.MDPChasing.reward import RewardFunctionCompete, RewardFunction
from src.MDPChasing.state import GetAgentPosFromState
from src.MDPChasing.policies import stationaryAgentPolicy, RandomPolicy
from src.algorithms.qLearning import GetAction, UpdateQTable, initQtable, argMax, QLearningAgent
from src.chooseFromDistribution import maxFromDistribution
from src.visualization import * 
from src.visualization import InitializeScreen,DrawBackground,DrawNewState, DrawText
from src.trajectory import SampleTrajectory
from src.writer import WriteDataFrameToCSV, loadFromPickle
from src.controller import ModelControllerMCTS
from src.simulationTrial import NormalTrialQLearningMaze
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


def qLearningPolicy(condition):
    # condition['isMutiGoal'] 

    renderOn = True
    numSimulations = 11
    maxRolloutSteps = 22
    maxRunningSteps = 33

    resultsPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'results'))

    numOfAgent = 5  # 1 hunter, 1 stag with high value, 3 rabbits with low value
    hunterId = [0]
    stagId = [1]
    rabbitId = [2, 3, 4]

    if condition['isMutiGoal'] :
        print(condition['trainingEpisodes'])
        targetIds = stagId + rabbitId
        episodes=condition['trainingEpisodes'] *4
        condition['trainingEpisodes']=condition['trainingEpisodes']*4

    else:

        targetIds = stagId
        episodes = condition['trainingEpisodes']

    positionIndex = [0, 1]
    getHunterPos_ = GetAgentPosFromState(hunterId, positionIndex)
    getHunterPos = lambda state: getHunterPos_(state[0])
    getTargetsPos_s = GetAgentPosFromState(targetIds, positionIndex)
    getTargetsPos = lambda state: getTargetsPos_s(state[0])
    getStaqPos_ = GetAgentPosFromState(stagId, positionIndex)
    getStaqPos = lambda state: getStaqPos_(state[0])

    lowerBound = 0
    gridSize = 15
    upperBound = [gridSize - 1, gridSize - 1]

    actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    numActionSpace = len(actionSpace)
    sheepSpeedRatio = 1
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * sheepSpeedRatio))


    stayWithinBoundary = env.StayWithinBoundaryMaze(upperBound, lowerBound)
    killZone=0
    isTerminal = env.IsTerminalWithKillZone(getHunterPos, getTargetsPos,killZone)
    # isTerminal = env.IsTerminal(getHunterPos, getTargetsPos)

    highValueSteps = 19
    numberOfMaze = 5
    loadPath = os.path.join('..', 'pathPlanning', 'map',
                            str(highValueSteps) + 'highValueSteps' + str(numberOfMaze) + 'maps' + '.pickle')
    mazeList = loadFromPickle(loadPath)
    if condition['fixAgentPostion']:
        reset = env.ResetMazeNoWallFix(upperBound, lowerBound, mazeList)
    
    else:
        reset = env.ResetMazeNoWallForTrain(upperBound, lowerBound, mazeList)

    expReset = env.ResetMazeNoWallFix(upperBound, lowerBound, mazeList)

    
    # fixReset = lambda: [(1, 1), (8, 8), (8, 9), (9, 9), (9, 8)]

    transitionFunction = env.TransitionWithObstacles(stayWithinBoundary)


    stagPolicy = RandomPolicy(sheepActionSpace)
    stagPolicy = stationaryAgentPolicy

    rabbitPolicies = [stationaryAgentPolicy] * len(rabbitId)

    def wolfTransit(state, action): return transitionFunction(
        state, [action, maxFromDistribution(stagPolicy(state))] + [maxFromDistribution(rabbitPolicy(state)) for rabbitPolicy in rabbitPolicies])

    maxRunningSteps = 100
    stepPenalty = -1 / maxRunningSteps
    catchBonus = 10
    highRewardRatio = 2
    rewardFunction = RewardFunction(highRewardRatio, stepPenalty, catchBonus, isTerminal,getHunterPos,getStaqPos)

# q-learing
    initQ = initQtable(actionSpace)
    discountFactor = 0.9
    learningRate = 0.01
    updateQTable = UpdateQTable(discountFactor, learningRate)
    epsilon = 0.1
    getAction = GetAction(epsilon, actionSpace, argMax)

    maxRunningStepsPerEpisodes = 100
    qLearningAgent=QLearningAgent(initQ, episodes, maxRunningStepsPerEpisodes, reset, wolfTransit, isTerminal, rewardFunction, updateQTable, getAction)
    startTime = time.time()
    QDict = qLearningAgent()
    finshedTime = time.time() - startTime
    print('time:', finshedTime)


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
    targetColor = [255, 250, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    textColorTuple = (255, 50, 50)


    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    drawText = DrawText(screen, drawBackground)

 

    for i in range(1):
        print(i)
        # expDesignValues = [[condition, diff] for condition in conditionList for diff in targetDiffsList] * numBlocks
        # random.shuffle(expDesignValues)
        # expDesignValues.append(specialDesign)

        modelController = ModelControllerMCTS(softmaxBeta)
        controller = modelController


        normalTrial = NormalTrialQLearningMaze(renderOn, controller, drawNewState, drawText,transitionFunction,isTerminal)

        experimentValues = co.OrderedDict()
        experimentValues["name"] = "fixReset{}isMutiGoal{}episodes{}step_agent".format(condition['fixAgentPostion'],condition['isMutiGoal'], episodes)
        
        # experimentValues["name"] = "maxRolloutSteps" + str(maxRolloutSteps) + '_' + "numSimulations" + str(numSimulations) +\
                                #    '_' + "softmaxBeta" + str(softmaxBeta) + '_' + "stepPenalty" + str(maxRunningSteps) + '_' + str(i)
        resultsDirPath = os.path.join(resultsPath,  "QLearningsoftmaxBeta" + str(softmaxBeta))
        if not os.path.exists(resultsDirPath):
            os.mkdir(resultsDirPath)

        writerPath = os.path.join(resultsDirPath, experimentValues["name"] + '.csv')
        writer = WriteDataFrameToCSV(writerPath)
        experiment = ModelSimulationMazeQLearning(expReset, normalTrial, writer, experimentValues, resultsPath)
        conditionList=range(100)
        experiment(conditionList,policy,condition)

class ModelSimulationMazeQLearning():
    def __init__(self, creatMap, normalTrial, writer, experimentValues, resultsPath):
        self.creatMap = creatMap
        self.normalTrial = normalTrial
        self.writer = writer
        self.experimentValues = experimentValues
        self.resultsPath = resultsPath

    def __call__(self, conditionList,policy,conditionInfo):
        for trialIndex, condition in enumerate(conditionList):
            initialState = self.creatMap()
            results = self.normalTrial(initialState, policy,conditionInfo)


            # results["conditionName"] = condition.name
            # results["decisionSteps"] = str(condition.decisionSteps)
            # results["obstacles"] = str(obstacles)



            response = self.experimentValues.copy()
            response.update(results)
            self.writer(response, trialIndex)

def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['isMutiGoal'] =[0,1]# [200, 400, 600]#[0.0, 1.0]
    manipulatedVariables['trainingEpisodes'] = [2000,4000,6000]#[20, 30]# [0.0, 0.2, 0.4]
    manipulatedVariables['fixAgentPostion'] = [0,1]#[10, 20, 30]
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditions = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    for condition in conditions:
        print(condition)
        qLearningPolicy(condition)
        # try:
        #     mctsPolicy(condition)
        # except:
        #     continue


if __name__ == "__main__":
    main()
