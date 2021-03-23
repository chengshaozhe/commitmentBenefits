import pandas as pd
import os
import glob
import itertools as it
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind
from collections import Counter,OrderedDict

# from dataAnalysis import calculateSE, calculateFirstIntentionStep


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    stdList = []
    participants = ['human', 'noise0.0673_softmaxBeta2.5', 'max']
    # participants = ['human']
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numSimulations'] = [200, 400, 600]#[0.0, 1.0]
    manipulatedVariables['maxRolloutSteps'] = [20, 30]# [0.0, 0.2, 0.4]
    manipulatedVariables['maxRunningSteps'] = [10, 20, 30]
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditions = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    dfTotal=pd.DataFrame(conditions)
    dfTotal['meanStep'] = [0]*len(conditions)

    # print(dfTotal)


    for i,condition in enumerate(conditions):
        softmaxBeta=-1
        numSimulations = condition['numSimulations']
        maxRolloutSteps = condition['maxRolloutSteps']
        maxRunningSteps = condition['maxRunningSteps']

        filename= "maxRolloutSteps" + str(maxRolloutSteps) + '_' + "numSimulations" + str(numSimulations) + \
                                   '_' + "softmaxBeta" + str(softmaxBeta) + '_' + "stepPenalty" + str(
            maxRunningSteps) + '_' + str(0)
        resultsDirPath = os.path.join(resultsPath, "softmaxBeta" + str(softmaxBeta))
        writerPath = os.path.join(resultsDirPath, filename + '.csv')
        df = pd.read_csv(writerPath)


        dfTotal['meanStep'][i] =np.mean(df['steps'])
    print(dfTotal)

    # for participant in participants:
    #     dataPath = os.path.join(resultsPath, participant)
    #     df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
    #
    #     df['firstIntentionStep'] = df.apply(lambda x: calculateFirstIntentionStep(eval(x['goal'])), axis=1)
    #     df['totalStep'] = df.apply(lambda x: len(eval(x['trajectory'])), axis=1)
    #
    #     # df.to_csv("all.csv")
    #     nubOfSubj = len(df["name"].unique())
    #     print(participant, nubOfSubj)
    #
    #     # dfExpTrail = df[(df['conditionName'] == 'expCondition2') & (df['decisionSteps'] == 6)]
    #
    #     dfExpTrail = df
    #
    #     # dfExpTrail = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] != 'special')]
    #     # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] != 'none')]
    #     # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'midLine')]
    #     # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'straightLine')]
    #     # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'straightLine') & (df['intentionedDisToTargetMin'] == 2)]
    #
    #     # dfExpTrail = df[(df['areaType'] == 'straightLine') | (df['areaType'] == 'midLine') & (df['distanceDiff'] == 0)]
    #     # dfExpTrail = df[(df['areaType'] != 'none')]
    #     # dfExpTrail = df[(df['areaType'] == 'expRect') & (df['areaType'] != 'rect')]
    #
    #     # dfExpTrail = df[df['noiseNumber'] != 'special']
    #     # dfExpTrail = df
    #
    #     statDF = pd.DataFrame()
    #     # statDF['firstIntentionStep'] = dfExpTrail.groupby('name')["firstIntentionStep"].mean()
    #     statDF['totalStep'] = dfExpTrail.groupby('name')["totalStep"].mean()
    #
    #     print('totalStep', np.mean(statDF['totalStep']))
    #     print('')
    #
    #     stats = statDF.columns
    #     statsList.append([np.mean(statDF[stat]) for stat in stats])
    #     stdList.append([calculateSE(statDF[stat]) for stat in stats])
    #
    # xlabels = ['totalStep']
    # labels = participants
    # x = np.arange(len(xlabels))
    # totalWidth, n = 0.1, len(participants)
    # width = totalWidth / n
    # x = x - (totalWidth - width) / 2
    # for i in range(len(statsList)):
    #     plt.bar(x + width * i, statsList[i], yerr=stdList[i], width=width, label=labels[i])
    # plt.xticks(x, xlabels)
    # # plt.ylim((0, 10))
    # plt.legend(loc='best')
    #
    # plt.title('firstIntentionStep')
    # plt.show()
