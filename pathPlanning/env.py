"""
Env 2D
@author: huiming zhou
"""

import numpy as np

class Env:
    def __init__(self):
        self.x_range = 15  # size of background
        self.y_range = 15
        self.randomObsNum=18
        # self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
        #                 (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.motions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        self.agentPos = [(7, 7)]
        self.goalPos = {(1, 1), (1, 14), (14, 1), (14, 14)}
        self.HighValueGoalPos = [(1, 1)]
        # self.HighValueGoalPos = [list(self.goalPos)[np.random.choice(range(len(self.goalPos)))]]
        self.lowValueGoalPos = list(self.goalPos-set(self.HighValueGoalPos))
        self.obs = self.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """

        x = self.x_range
        y = self.y_range
        obs = set()

        for i in range(x):
            obs.add((i, -1))
        for i in range(x):
            obs.add((i, y ))

        for i in range(y):
            obs.add((-1, i))
        for i in range(y):
            obs.add((x , i))
        np.random.seed(122)
        allGround = list(set([(x, y) for x in range(1,self.x_range-1) for y in range(1,self.y_range-1)])-self.goalPos-set(self.agentPos))
        obsIdList = np.random.choice(range(len(allGround)),self.randomObsNum,False)
        print(obsIdList)
        for i in range(self.randomObsNum):
            obs.add(allGround[obsIdList[i]])

        obs.add((1, 2))
        obs.add((2, 2))
        obs.add((3, 2))
        # for i in range(10, 21):
        #     obs.add((i, 15))
        # for i in range(15):
        #     obs.add((20, i))
        #
        # for i in range(15, 30):
        #     obs.add((30, i))
        # for i in range(16):
        #     obs.add((40, i))

        return obs