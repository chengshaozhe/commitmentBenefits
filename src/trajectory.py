import numpy as np
import random


class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, resetState, chooseAction, renderOn, viz, resetPolicy=None):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.resetState = resetState
        self.chooseAction = chooseAction
        self.renderOn = renderOn
        self.viz = viz
        self.resetPolicy = resetPolicy

    def __call__(self, policy):

        state = self.resetState()
        while self.isTerminal(state):
            state = self.resetState()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break

            # need to clean
            if self.renderOn:
                playerPosition = state[0]
                targetPositions = state[1:]
                obstacles = []
                self.viz(playerPosition, targetPositions, obstacles)

            actionDists = policy(state)
            action = [choose(actionDist) for choose, actionDist in zip(self.chooseAction, actionDists)]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            state = nextState

        if self.resetPolicy:
            policyAttributes = self.resetPolicy()
            if policyAttributes:
                trajectoryWithPolicyAttrVals = [tuple(list(stateActionPair) + list(policyAttribute))
                                                for stateActionPair, policyAttribute in zip(trajectory, policyAttributes)]
                trajectory = trajectoryWithPolicyAttrVals.copy()
        return trajectory

class SampleTrajectoryMaze:
    def __init__(self, maxRunningSteps, transit, isTerminal, resetState, chooseAction, renderOn, viz, resetPolicy=None):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.resetState = resetState
        self.chooseAction = chooseAction
        self.renderOn = renderOn
        self.viz = viz
        self.resetPolicy = resetPolicy

    def __call__(self, policy):

        state = self.resetState()
        while self.isTerminal(state):
            state = self.resetState()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break

            # need to clean
            if self.renderOn:
                playerPosition = state[0][0]
                targetPositions = state[0][1:]
                obstacles = state[1]
                self.viz(playerPosition, targetPositions, obstacles)

            actionDists = policy(state)
            action = [choose(actionDist) for choose, actionDist in zip(self.chooseAction, actionDists)]
            trajectory.append((state, action, actionDists))
            print(state[0])
            print(action)
            nextState = self.transit(state, action )

            state = nextState

        if self.resetPolicy:
            policyAttributes = self.resetPolicy()
            if policyAttributes:
                trajectoryWithPolicyAttrVals = [tuple(list(stateActionPair) + list(policyAttribute))
                                                for stateActionPair, policyAttribute in zip(trajectory, policyAttributes)]
                trajectory = trajectoryWithPolicyAttrVals.copy()
        return trajectory