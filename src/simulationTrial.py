import numpy as np
import pygame as pg
from pygame import time
import collections as co
import pickle
import random


def calculateGridDis(grid1, grid2):
    gridDis = np.linalg.norm(np.array(grid1) - np.array(grid2), ord=1)
    return int(gridDis)


def creatRect(coor1, coor2):
    vector = np.array(list(zip(coor1, coor2)))
    vector.sort(axis=1)
    rect = [(i, j) for i in range(vector[0][0], vector[0][1] + 1) for j in range(vector[1][0], vector[1][1] + 1)]
    return rect


def calculateAvoidCommitmnetZone(playerGrid, target1, target2):
    dis1 = calculateGridDis(playerGrid, target1)
    dis2 = calculateGridDis(playerGrid, target2)
    if dis1 == dis2:
        rect1 = creatRect(playerGrid, target1)
        rect2 = creatRect(playerGrid, target2)
        avoidCommitmentZone = list(set(rect1).intersection(set(rect2)))
        avoidCommitmentZone.remove(tuple(playerGrid))
    else:
        avoidCommitmentZone = []

    return avoidCommitmentZone


def calculateAvoidCommitmnetZoneAll(playerGrid, target1, target2):
    rect1 = creatRect(playerGrid, target1)
    rect2 = creatRect(playerGrid, target2)
    avoidCommitmentZone = list(set(rect1).intersection(set(rect2)))
    avoidCommitmentZone.remove(tuple(playerGrid))
    return avoidCommitmentZone


def calMidPoints(initPlayerGrid, target1, target2):
    zone = calculateAvoidCommitmnetZoneAll(initPlayerGrid, target1, target2)
    if zone:
        playerDisToZoneGrid = [calculateGridDis(initPlayerGrid, zoneGrid) for zoneGrid in zone]
        midPointIndex = np.argmax(playerDisToZoneGrid)
        midPoint = zone[midPointIndex]
    else:
        midPoint = initPlayerGrid
    return midPoint


def inferGoal(originGrid, aimGrid, targetGridA, targetGridB):
    pacmanBean1aimDisplacement = calculateGridDis(targetGridA, aimGrid)
    pacmanBean2aimDisplacement = calculateGridDis(targetGridB, aimGrid)
    pacmanBean1LastStepDisplacement = calculateGridDis(targetGridA, originGrid)
    pacmanBean2LastStepDisplacement = calculateGridDis(targetGridB, originGrid)
    bean1Goal = pacmanBean1LastStepDisplacement - pacmanBean1aimDisplacement
    bean2Goal = pacmanBean2LastStepDisplacement - pacmanBean2aimDisplacement
    if bean1Goal > bean2Goal:
        goal = 1
    elif bean1Goal < bean2Goal:
        goal = 2
    else:
        goal = 0
    return goal


def checkTerminationOfTrial(bean1Grid, bean2Grid, humanGrid):
    if calculateGridDis(humanGrid, bean1Grid) == 0 or calculateGridDis(humanGrid, bean2Grid) == 0:
        pause = False
    else:
        pause = True
    return pause

class NormalTrialQLearningMaze():
    def __init__(self, renderOn, controller, drawNewState, drawText, transit, isTerminal):
        self.renderOn = renderOn
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.transit = transit
        self.isTerminal = isTerminal
    def __call__(self, initialState, policy,condition):
        state=initialState
        initialTime = time.get_ticks()
        reactionTime = list()
        initialPlayerGrid = state[0][0]
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimPlayerGridList = []
        stepCount = 0

        pause = True
        while pause:
            if self.renderOn:
                playerPosition = state[0][0]

                targetPositions = state[0][1:]
                obstacles = state[1]
                self.drawNewState(playerPosition, targetPositions, obstacles)
            aimAction = self.controller(state, policy)

            state = self.transit(state, aimAction)
            stepCount = stepCount + 1
            realPlayerGrid = state[0][0]
            reactionTime.append(time.get_ticks() - initialTime)
            trajectory.append(list(realPlayerGrid))
            pause =  not self.isTerminal(state)
        

        results["isMutiGoal"] = str(condition['isMutiGoal'])
        results["trainingEpisodes"] = str(condition['trainingEpisodes'])
        results["fixAgentPostion"] = str(condition['fixAgentPostion'])

        results["trajectory"] = str(trajectory)
        results["lastPos"] = str(trajectory[-1])
        results["steps"] = str(len(trajectory))

        print(len(trajectory))
        return results

class NormalTrialMCTSMaze():
    def __init__(self, renderOn, controller, drawNewState, drawText, transit, isTerminal):
        self.renderOn = renderOn
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.transit = transit
        self.isTerminal = isTerminal
    def __call__(self, initialState, MCTS):
        state=initialState
        initialTime = time.get_ticks()
        reactionTime = list()
        initialPlayerGrid = state[0][0]
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimPlayerGridList = []
        stepCount = 0

        pause = True
        while pause:
            if self.renderOn:
                playerPosition = state[0][0]

                targetPositions = state[0][1:]
                obstacles = state[1]
                self.drawNewState(playerPosition, targetPositions, obstacles)
            aimAction = self.controller(state, MCTS)

            state = self.transit(state, aimAction)
            stepCount = stepCount + 1
            realPlayerGrid = state[0][0]
            reactionTime.append(time.get_ticks() - initialTime)
            trajectory.append(list(realPlayerGrid))
            pause =  not self.isTerminal(state)
        results["trajectory"] = str(trajectory)
        results["steps"] = str(len(trajectory))
        print(len(trajectory))
        return results
class NormalTrial():
    def __init__(self, renderOn, controller, drawNewState, drawText, normalNoise, checkBoundary):
        self.renderOn = renderOn
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.normalNoise = normalNoise
        self.checkBoundary = checkBoundary

    def __call__(self, bean1Grid, bean2Grid, playerGrid, obstacles, designValues, QDict):
        initialPlayerGrid = tuple(playerGrid)
        initialTime = time.get_ticks()
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        aimPlayerGridList = []
        leastStep = min([calculateGridDis(playerGrid, beanGrid) for beanGrid in [bean1Grid, bean2Grid]])
        noiseStep = sorted(random.sample(list(range(2, leastStep)), designValues))
        stepCount = 0
        goalList = list()

        realPlayerGrid = initialPlayerGrid
        pause = True
        while pause:
            if self.renderOn:
                self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid, obstacles)
            aimPlayerGrid, aimAction = self.controller(realPlayerGrid, bean1Grid, bean2Grid, QDict)
            goal = inferGoal(realPlayerGrid, aimPlayerGrid, bean1Grid, bean2Grid)
            goalList.append(goal)
            stepCount = stepCount + 1
            noisePlayerGrid, realAction = self.normalNoise(realPlayerGrid, aimAction, noiseStep, stepCount)
            if noisePlayerGrid in obstacles:
                noisePlayerGrid = tuple(trajectory[-1])
            realPlayerGrid = self.checkBoundary(noisePlayerGrid)
            reactionTime.append(time.get_ticks() - initialTime)
            trajectory.append(list(realPlayerGrid))
            aimActionList.append(aimAction)
            aimPlayerGridList.append(aimPlayerGrid)
            pause = checkTerminationOfTrial(bean1Grid, bean2Grid, realPlayerGrid)
        results["reactionTime"] = str(reactionTime)
        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(aimActionList)
        results["aimPlayerGridList"] = str(aimPlayerGridList)
        results["noisePoint"] = str(noiseStep)
        results["goal"] = str(goalList)
        return results


class SpecialTrial():
    def __init__(self, renderOn, controller, drawNewState, drawText, specialNoise, checkBoundary):
        self.renderOn = renderOn
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.specialNoise = specialNoise
        self.checkBoundary = checkBoundary

    def __call__(self, bean1Grid, bean2Grid, playerGrid, obstacles, QDict):
        initialPlayerGrid = tuple(playerGrid)
        initialTime = time.get_ticks()
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        aimPlayerGridList = []
        firstIntentionFlag = False
        noiseStep = list()
        stepCount = 0
        goalList = list()

        pause = True
        realPlayerGrid = initialPlayerGrid
        while pause:
            if self.renderOn:
                self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid, obstacles)
            aimPlayerGrid, aimAction = self.controller(realPlayerGrid, bean1Grid, bean2Grid, QDict)
            goal = inferGoal(realPlayerGrid, aimPlayerGrid, bean1Grid, bean2Grid)
            goalList.append(goal)
            stepCount = stepCount + 1

            if len(trajectory) > 3:
                noisePlayerGrid, noiseStep, firstIntentionFlag = self.specialNoise(trajectory, bean1Grid, bean2Grid, noiseStep, firstIntentionFlag)
                if noisePlayerGrid:
                    realPlayerGrid = self.checkBoundary(noisePlayerGrid)
                else:
                    realPlayerGrid = self.checkBoundary(aimPlayerGrid)
            else:
                realPlayerGrid = self.checkBoundary(aimPlayerGrid)
            if realPlayerGrid in obstacles:
                realPlayerGrid = tuple(trajectory[-1])
            reactionTime.append(time.get_ticks() - initialTime)
            trajectory.append(list(realPlayerGrid))
            aimActionList.append(aimAction)
            aimPlayerGridList.append(aimPlayerGrid)
            pause = checkTerminationOfTrial(bean1Grid, bean2Grid, realPlayerGrid)

        results["reactionTime"] = str(reactionTime)
        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(aimActionList)
        results["aimPlayerGridList"] = str(aimPlayerGridList)
        results["noisePoint"] = str(noiseStep)
        results["goal"] = str(goalList)
        return results


class NormalTrialOnline():
    def __init__(self, renderOn, controller, drawNewState, drawText, normalNoise, checkBoundary):
        self.renderOn = renderOn
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.normalNoise = normalNoise
        self.checkBoundary = checkBoundary

    def __call__(self, Q_dictList, bean1Grid, bean2Grid, playerGrid, obstacles, designValues):
        initialPlayerGrid = tuple(playerGrid)
        initialTime = time.get_ticks()
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        aimPlayerGridList = []
        leastStep = min([calculateGridDis(playerGrid, beanGrid) for beanGrid in [bean1Grid, bean2Grid]])
        noiseStep = sorted(random.sample(list(range(2, leastStep)), designValues))
        stepCount = 0
        goalList = list()

        midpoint = calMidPoints(initialPlayerGrid, bean1Grid, bean2Grid)
        disToMidPoint = calculateGridDis(initialPlayerGrid, midpoint)
        avoidCommitQDicts, commitQDicts = Q_dictList
        target = chooseGoal(playerGrid, bean1Grid, bean2Grid)

        realPlayerGrid = initialPlayerGrid
        pause = True
        while pause:
            if self.renderOn:
                self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid, obstacles)

            commited = 1
            if commited:
                Q_dict = commitQDicts[target]
            else:
                target = random.choice([bean1Grid, bean2Grid])
                Q_dict = avoidCommitQDicts[target]

            aimPlayerGrid, aimAction = self.controller(Q_dict, realPlayerGrid)

            goal = inferGoal(realPlayerGrid, aimPlayerGrid, bean1Grid, bean2Grid)
            goalList.append(goal)
            stepCount = stepCount + 1
            noisePlayerGrid, realAction = self.normalNoise(realPlayerGrid, aimAction, noiseStep, stepCount)
            if noisePlayerGrid in obstacles:
                noisePlayerGrid = tuple(trajectory[-1])
            realPlayerGrid = self.checkBoundary(noisePlayerGrid)
            reactionTime.append(time.get_ticks() - initialTime)
            trajectory.append(list(realPlayerGrid))
            aimActionList.append(aimAction)
            aimPlayerGridList.append(aimPlayerGrid)
            pause = checkTerminationOfTrial(bean1Grid, bean2Grid, realPlayerGrid)
        results["reactionTime"] = str(reactionTime)
        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(aimActionList)
        results["aimPlayerGridList"] = str(aimPlayerGridList)
        results["noisePoint"] = str(noiseStep)
        results["goal"] = str(goalList)
        return results


class SpecialTrialOnline():
    def __init__(self, renderOn, controller, drawNewState, drawText, specialNoise, checkBoundary):
        self.renderOn = renderOn
        self.controller = controller
        self.drawNewState = drawNewState
        self.drawText = drawText
        self.specialNoise = specialNoise
        self.checkBoundary = checkBoundary

    def __call__(self, Q_dictList, bean1Grid, bean2Grid, playerGrid, obstacles):
        initialPlayerGrid = tuple(playerGrid)
        initialTime = time.get_ticks()
        reactionTime = list()
        trajectory = [initialPlayerGrid]
        results = co.OrderedDict()
        aimActionList = list()
        aimPlayerGridList = []
        firstIntentionFlag = False
        noiseStep = list()
        stepCount = 0
        goalList = list()

        midpoint = calMidPoints(initialPlayerGrid, bean1Grid, bean2Grid)
        disToMidPoint = calculateGridDis(initialPlayerGrid, midpoint)
        avoidCommitQDicts, commitQDicts = Q_dictList
        target = chooseGoal(playerGrid, bean1Grid, bean2Grid)

        pause = True
        realPlayerGrid = initialPlayerGrid
        while pause:
            if self.renderOn:
                self.drawNewState(bean1Grid, bean2Grid, realPlayerGrid, obstacles)

            commited = 1
            if commited:
                Q_dict = commitQDicts[target]
            else:
                target = random.choice([bean1Grid, bean2Grid])
                Q_dict = avoidCommitQDicts[target]

            aimPlayerGrid, aimAction = self.controller(Q_dict, realPlayerGrid)

            goal = inferGoal(realPlayerGrid, aimPlayerGrid, bean1Grid, bean2Grid)
            goalList.append(goal)
            stepCount = stepCount + 1

            if len(trajectory) > 3:
                noisePlayerGrid, noiseStep, firstIntentionFlag = self.specialNoise(trajectory, bean1Grid, bean2Grid, noiseStep, firstIntentionFlag)
                if noisePlayerGrid:
                    realPlayerGrid = self.checkBoundary(noisePlayerGrid)
                else:
                    realPlayerGrid = self.checkBoundary(aimPlayerGrid)
            else:
                realPlayerGrid = self.checkBoundary(aimPlayerGrid)
            if realPlayerGrid in obstacles:
                realPlayerGrid = tuple(trajectory[-1])
            reactionTime.append(time.get_ticks() - initialTime)
            trajectory.append(list(realPlayerGrid))
            aimActionList.append(aimAction)
            aimPlayerGridList.append(aimPlayerGrid)
            pause = checkTerminationOfTrial(bean1Grid, bean2Grid, realPlayerGrid)

        results["reactionTime"] = str(reactionTime)
        results["trajectory"] = str(trajectory)
        results["aimAction"] = str(aimActionList)
        results["aimPlayerGridList"] = str(aimPlayerGridList)
        results["noisePoint"] = str(noiseStep)
        results["goal"] = str(goalList)
        return results
