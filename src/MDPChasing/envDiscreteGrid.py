import numpy as np
from random import randint
from random import  sample

class Reset:
    def __init__(self, gridSize, lowerBound, agentCount):
        self.gridX, self.gridY = gridSize
        self.lowerBound = lowerBound
        self.agentCount = agentCount

    def __call__(self):
        startState = [(randint(self.lowerBound, self.gridX), randint(self.lowerBound, self.gridY)) for _ in range(self.agentCount)]

        return startState
class FixReset:
    def __init__(self, agentState, stagState, rabbitState):
        self.agentState = agentState
        self.stagState = stagState
        self.rabbitState = rabbitState

    def __call__(self):
        startState = [self.agentState] + [self.stagState] + [list(state) for state in  self.rabbitState]
        return startState

class ResetMazeNoWallForTrain:
    def __init__(self, gridSize, lowerBound,MazeList):
        self.gridX, self.gridY = gridSize
        self.lowerBound = lowerBound
        self.MazeList = MazeList

    def __call__(self):
        # maze = np.random.sample(self.MazeList)
        maze = sample(self.MazeList, 1)[0]
        # agentState = list(maze['initAgent'])
        agentState = list((randint(self.lowerBound, self.gridX), randint(self.lowerBound, self.gridY)))  
        stagState = list(maze['highValueGoalPos'])
        rabbitState = list(maze['lowValueGoalsPos'])
        startState = [agentState]+[stagState]+[list(state) for state in rabbitState]
        # obs = maze['fixedObstacles']
        obs=[]
        return [startState, obs]

class ResetMazeNoWallFix:
    def __init__(self, gridSize, lowerBound,MazeList):
        self.gridX, self.gridY = gridSize
        self.lowerBound = lowerBound
        self.MazeList = MazeList

    def __call__(self):
        # maze = np.random.sample(self.MazeList)
        maze = sample(self.MazeList, 1)[0]
        agentState = list(maze['initAgent'])
        # agentState = list((randint(self.lowerBound, self.gridX), randint(self.lowerBound, self.gridY)))  
        stagState = list(maze['highValueGoalPos'])
        rabbitState = list(maze['lowValueGoalsPos'])
        startState = [agentState]+[stagState]+[list(state) for state in rabbitState]
        # obs = maze['fixedObstacles']
        obs=[]
        return [startState, obs]
        
class ResetMaze:
    def __init__(self, gridSize, lowerBound,MazeList):
        self.gridX, self.gridY = gridSize
        self.lowerBound = lowerBound
        self.MazeList = MazeList

    def __call__(self):
        # maze = np.random.sample(self.MazeList)
        maze = sample(self.MazeList, 1)[0]
        agentState = list(maze['initAgent'])
        stagState = list(maze['highValueGoalPos'])
        rabbitState = list(maze['lowValueGoalsPos'])
        startState = [agentState]+[stagState]+[list(state) for state in rabbitState]
        obs = maze['fixedObstacles']
        # obs=[]
        return [startState, obs]

class StayWithinBoundary:
    def __init__(self, gridSize, lowerBoundary):
        self.gridX, self.gridY = gridSize
        self.lowerBoundary = lowerBoundary

    def __call__(self, nextIntendedState):
        nextX, nextY = nextIntendedState
        if nextX < self.lowerBoundary:
            nextX = self.lowerBoundary
        if nextX > self.gridX:
            nextX = self.gridX
        if nextY < self.lowerBoundary:
            nextY = self.lowerBoundary
        if nextY > self.gridY:
            nextY = self.gridY
        return nextX, nextY

class StayWithinBoundaryMaze:
    def __init__(self, gridSize, lowerBoundary):
        self.gridX, self.gridY = gridSize
        self.lowerBoundary = lowerBoundary

    def __call__(self, state,action,obstacleList):

        nextIntendedState = np.array(state)+np.array(action)
        nextX, nextY = nextIntendedState
        if nextX < self.lowerBoundary:
            nextX = self.lowerBoundary
        if nextX > self.gridX:
            nextX = self.gridX
        if nextY < self.lowerBoundary:
            nextY = self.lowerBoundary
        if nextY > self.gridY:
            nextY = self.gridY

        if (nextX, nextY) in obstacleList:
            nextX, nextY = np.array(state)
        return nextX, nextY

class IsHitObstacles:
    def __init__(self, obstacles):
        self.obstacles = obstacles

    def __call__(self, nextIntendedState):
        if nextIntendedState in self.obstacles:
            return True
        else:
            return False


class TransitionWithObstacles:
    def __init__(self, stayWithinBoundaryMaze):
        self.stayWithinBoundaryMaze = stayWithinBoundaryMaze

    def __call__(self, stateList,actionList):
        # print(stateList,actionList)
        obstacleList = stateList[1]
        agentStateList = stateList[0]
        # [print(state,  list(action), obstacleList) for state,action in zip(agentStateList,actionList)]
        # for state, action in zip(agentStateList, actionList):

        agentsNextState = [self.stayWithinBoundaryMaze(state, action, obstacleList) for state,action in zip(agentStateList,actionList)]
        return [agentsNextState,obstacleList]


class Transition:
    def __init__(self, stayWithinBoundary):
        self.stayWithinBoundary = stayWithinBoundary

    def __call__(self, actionList, state):
        agentsIntendedState = np.array(state) + np.array(actionList)
        agentsNextState = [self.stayWithinBoundary(intendedState) for intendedState in agentsIntendedState]
        return agentsNextState

class IsTerminalWithKillZone():
    def __init__(self, locatePredator, locatePrey, minDistance):
        self.locatePredator = locatePredator
        self.locatePrey = locatePrey
        self.minDistance = minDistance

    def __call__(self, state):
        terminal = False
        predatorPosition = self.locatePredator(state)
        preyPositions = self.locatePrey(state)

        L1Normdistances = [np.linalg.norm((np.array(preyPosition) - np.array(predatorPosition)), ord=1) for preyPosition in preyPositions ]
        if np.any(np.array(L1Normdistances) <= self.minDistance):
            terminal = True
        return terminal
class IsTerminal:
    def __init__(self, locatePredator, locatePrey):
        self.locatePredator = locatePredator
        self.locatePrey = locatePrey

    def __call__(self, state):
        predatorPosition = self.locatePredator(state)
        preyPositions = self.locatePrey(state)

        isAnyTerminal = [np.all(np.array(predatorPosition) == np.array(preyPosition)) for preyPosition in preyPositions]
        if np.any(isAnyTerminal):
            return True
        else:
            return False
