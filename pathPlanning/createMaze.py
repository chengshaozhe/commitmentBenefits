import os
import sys
import math
import heapq
import numpy as np
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__))))

import plotting
import env
import Astar
import pickle
from collections import namedtuple


def loadFromPickle(path):
    pickleIn = open(path, 'rb')
    object = pickle.load(pickleIn)
    pickleIn.close()
    return object


def saveToPickle(data, path):
    pklFile = open(path, "wb")
    pickle.dump(data, pklFile)
    pklFile.close()


def calculateGridDis(grid1, grid2):
    gridDis = np.linalg.norm(np.array(grid1) - np.array(grid2), ord=1)
    return gridDis


class RandomSampleMaze:
    def __init__(self, x_range, y_range, randomObsNum, agentPos, HighValueGoalPos, lowValueGoalPos):
        self.x_range = x_range  # size of background
        self.y_range = y_range
        self.randomObsNum = randomObsNum
        self.agentPos = agentPos
        self.goalPos = set(HighValueGoalPos + lowValueGoalPos)
        # self.HighValueGoalPos = HighValueGoalPos
        # self.HighValueGoalPos = [list(self.goalPos)[np.random.choice(range(len(self.goalPos)))]]
        # self.lowValueGoalPos = list(self.goalPos-set(self.HighValueGoalPos))
        # self.lowValueGoalPos =  lowValueGoalPos

    def __call__(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """

        # x = self.x_range
        # y = self.y_range
        obs = set()
        # for i in range(x):
        #     obs.add((i, -1))
        # for i in range(x):
        #     obs.add((i, y ))
        # for i in range(y):
        #     obs.add((-1, i))
        # for i in range(y):
        #     obs.add((x , i))
        allGround = list(set([(x, y) for x in range(1, self.x_range - 1) for y in range(1, self.y_range - 1)]) - self.goalPos - set(self.agentPos))
        obsIdList = np.random.choice(range(len(allGround)), self.randomObsNum, False)
        for i in range(self.randomObsNum):
            obs.add(allGround[obsIdList[i]])

        return obs


def addWall(x, y, obs):
    for i in range(x):
        obs.add((i, -1))
    for i in range(x):
        obs.add((i, y))
    for i in range(y):
        obs.add((-1, i))
    for i in range(y):
        obs.add((x, i))
    return obs


def main():
    x_range = 15  # size of background
    y_range = 15
    s_start = (7, 7)
    highvalueGoal = (1, 13)
    lowvalueGoal = [(1, 1), (13, 1), (13, 13)]
    motions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    x_range = 15  # size of background
    y_range = 15
    randomObsNum = 18
    randomSampleMaze = RandomSampleMaze(x_range, y_range, randomObsNum, [s_start], [highvalueGoal], lowvalueGoal)

    maze = namedtuple('maze', 'initAgent highValueGoalPos lowValueGoalsPos highValueSteps fixedObstacles ')

    highValueSteps = 19
    lowValueSteps = 13
    numberOfMaze = 5

    # Samples = 1000000
    # for i in range(Samples):
    i = 0
    selectedMazeList = []

    visualize = True
    while i < numberOfMaze:
        proposalMaze = randomSampleMaze()
        proposalMazeWithWall = addWall(x_range, y_range, proposalMaze)
        astar = Astar.AStar(s_start, highvalueGoal, proposalMazeWithWall, motions, "manhattan")
        try:
            path = []
            path, visited = astar.searching()
        except:
            pass
        else:
            pass
        if len(path) == highValueSteps:
            # print(len(path))
            popOut = False
            for goal in lowvalueGoal:
                astar2 = Astar.AStar(s_start, goal, proposalMazeWithWall, motions, "manhattan")
                try:
                    path2, visited2 = astar2.searching()
                except:
                    popOut = True
                    break
                else:
                    # print(goal, len(path2))
                    if len(path2) != lowValueSteps:
                        popOut = True
                        break
            if popOut:
                continue

            minDistractDistance = []
            for goal in lowvalueGoal:

                minDistance = np.min([calculateGridDis(goal, node) for node in path])
                # print(minDistance)
                minDistractDistance.append({goal: minDistance})

            # currentMaze = maze(initAgent = s_start, highValueGoalPos = highvalueGoal, highValueSteps = len(path), fixedObstacles = proposalMaze)
            currentMaze = {'initAgent': s_start, 'highValueGoalPos': highvalueGoal, 'lowValueGoalsPos': lowvalueGoal, 'highValueSteps': len(path), 'distractDistance': minDistractDistance, 'fixedObstacles': proposalMaze}
            # print(currentMaze,i)
            i = i + 1
            print(currentMaze, i)
            selectedMazeList.append(currentMaze)
            if visualize:
                plot = plotting.Plotting(s_start, highvalueGoal, proposalMazeWithWall)
                plot.animation(path, visited, "A*")  # animation
#
    savePath = os.path.join('map', str(highValueSteps) + 'highValueSteps' + str(numberOfMaze) + 'maps' + '.pickle')
    saveToPickle(selectedMazeList, savePath)


if __name__ == '__main__':
    main()
