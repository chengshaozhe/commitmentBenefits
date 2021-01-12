import numpy as np
from random import randint


class Reset:
    def __init__(self, gridSize, lowerBound, agentCount):
        self.gridX, self.gridY = gridSize
        self.lowerBound = lowerBound
        self.agentCount = agentCount

    def __call__(self):
        startState = [(randint(self.lowerBound, self.gridX), randint(self.lowerBound, self.gridY)) for _ in range(self.agentCount)]
        return startState


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


class IsHitObstacles:
    def __init__(self, obstacles):
        self.obstacles = obstacles

    def __call__(self, nextIntendedState):
        if nextIntendedState in self.obstacles:
            return True
        else:
            return False


class TransitionWithObstacles:
    def __init__(self, stayWithinBoundary):
        self.stayWithinBoundary = stayWithinBoundary

    def __call__(self, actionList, state):
        agentsIntendedState = np.array(state) + np.array(actionList)
        agentsNextState = [self.stayWithinBoundary(intendedState) for intendedState in agentsIntendedState]
        return agentsNextState


class Transition:
    def __init__(self, stayWithinBoundary):
        self.stayWithinBoundary = stayWithinBoundary

    def __call__(self, actionList, state):
        agentsIntendedState = np.array(state) + np.array(actionList)
        agentsNextState = [self.stayWithinBoundary(intendedState) for intendedState in agentsIntendedState]
        return agentsNextState


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
