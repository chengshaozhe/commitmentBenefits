import numpy as np


class RewardFunction():
    def __init__(self, highRewardRatio, aliveBonus, deathPenalty, isTerminal,getHunterPos,getStaqPos):
        self.highRewardRatio = highRewardRatio
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal
        self.getHunterPos = getHunterPos
        self.getStaqPos = getStaqPos
    def __call__(self, state, action):
        reward = self.aliveBonus
        if self.isTerminal(state):
            # need to clean
            if np.all(self.getHunterPos(state) == self.getStaqPos(state)):
                reward += self.highRewardRatio * self.deathPenalty
                # print(reward)
            else:
                reward += self.deathPenalty *0
                # print(reward)

        return reward


class RewardFunctionCompete():
    def __init__(self, aliveBonus, deathPenalty, isTerminal):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal

    def __call__(self, state, action):
        reward = self.aliveBonus
        if self.isTerminal(state):
            reward += self.deathPenalty

        return reward


class IsCollided:
    def __init__(self, minXDis, getSelfPos, getOtherPos):
        self.minXDis = minXDis
        self.getSelfPos = getSelfPos
        self.getOtherPos = getOtherPos

    def __call__(self, state):
        state = np.asarray(state)
        selfPos = self.getSelfPos(state)
        otherPositions = [getPos(state) for getPos in self.getOtherPos]

        L2Normdistance = [np.linalg.norm((selfPos - otherPosition), ord=2) for otherPosition in otherPositions]
        terminal = np.any(np.array(L2Normdistance) <= self.minXDis)

        return terminal


class RewardFunctionWithWall():
    def __init__(self, aliveBonus, deathPenalty, safeBound, wallDisToCenter, wallPunishRatio, isTerminal, getPosition):
        self.aliveBonus = aliveBonus
        self.deathPenalty = deathPenalty
        self.safeBound = safeBound
        self.wallDisToCenter = wallDisToCenter
        self.wallPunishRatio = wallPunishRatio
        self.isTerminal = isTerminal
        self.getPosition = getPosition

    def __call__(self, state, action):
        reward = self.aliveBonus
        if self.isTerminal(state):
            reward += self.deathPenalty

        agentPos = self.getPosition(state)
        minDisToWall = np.min(np.array([np.abs(agentPos - self.wallDisToCenter), np.abs(agentPos + self.wallDisToCenter)]).flatten())
        wallPunish = - self.wallPunishRatio * np.abs(self.aliveBonus) * np.power(max(0, self.safeBound - minDisToWall), 2) / np.power(self.safeBound, 2)

        return reward + wallPunish


class HeuristicDistanceToTarget:
    def __init__(self, weight, getPredatorPosition, getPreyPosition):
        self.weight = weight
        self.getPredatorPosition = getPredatorPosition
        self.getPreyPosition = getPreyPosition

    def __call__(self, state):
        predatorPos = self.getPredatorPosition(state)
        preyPositions = self.getPreyPosition(state)
        distances = [np.linalg.norm(predatorPos - preyPos, ord=1) for preyPos in preyPositions]
        reward = -self.weight * np.min(distances)
        return reward
