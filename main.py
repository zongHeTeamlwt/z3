import warnings

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")  # 忽略警告信息

def dataPreview():
    # 数据集基本信息
    print(trainData.info())
    # 数字型特征的描述性统计
    print(trainData.describe())
    # 训练集缺失值检查
    print(trainData.isnull().sum())
    # 测试集缺失值检查
    print(testData.isnull().sum())


def analysisDataByPicture():
    # age
    trainData["Age"].hist(bins=50)
    plt.title("Age Analysis")
    plt.xlabel("Age")
    plt.ylabel("Frequency")

    g = sns.FacetGrid(trainData, col='Survived')
    g.map(plt.hist, 'Age', bins=20)

    grid = sns.FacetGrid(trainData, col='Pclass', hue='Survived')
    grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
    grid.add_legend()
    grid = sns.FacetGrid(trainData, col='Embarked')
    grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
    grid.add_legend()
    grid = sns.FacetGrid(trainData, col='Embarked', hue='Survived', palette={0: 'b', 1: 'r'})
    grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
    grid.add_legend()

    grid = sns.FacetGrid(trainData, col='Pclass', hue='Sex')
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend()



def fillByAverage(col):
    trainData[col].fillna(trainData[col].mean(), inplace=True)
    testData[col].fillna(trainData[col].mean(), inplace=True)


def fillByMost(col):
    trainData[col].fillna(trainData[col].mode()[0], inplace=True)
    testData[col].fillna(trainData[col].mode()[0], inplace=True)


def fillByUnknown(col):
    trainData[col].fillna("Unknown", inplace=True)
    testData[col].fillna("Unknown", inplace=True)


def classifierByZero_One(col, *args):
    trainData[col] = trainData[col].map({args[0]: 0, args[1]: 1})
    testData[col] = testData[col].map({args[0]: 0, args[1]: 1})


def one_hot(col, train, test):
    trainData = pd.get_dummies(train, columns=[col])
    testData = pd.get_dummies(test, columns=[col])
    return trainData, testData

def save(path, testData, yPred):
    submission = pd.DataFrame({'PassengerId': testData['PassengerId'], 'Survived': yPred})
    submission.to_csv(path, index=False)


if __name__ == "__main__":
    trainData = pd.read_csv("./titanic/train.csv")
    testData = pd.read_csv("./titanic/test.csv")

    # 数据分析以及预处理
    dataPreview()  # 数据查看
    # analysisDataByPicture()  # 数据分析（通过画图）

    fillByAverage("Age")  # 对年龄填充平均值
    fillByMost("Embarked")  # 登船港口，有两个缺失值，数量较少，用出现频率最多的填充即可
    fillByUnknown("Cabin")  # 填充夹板号，有大量缺失值，通常不会尝试去预测或估计具体的船舱号，也不能删除这个数据，因为其他的字段是有用的，这里我们把它表示为unknown
    fillByAverage("Fare")  # 填充票价，平均值

    #特征选择和特征工程
    # 2.1 sex: 转换为0-1，embarked: 使用one-hot编码
    classifierByZero_One("Sex", "male", "female")
    trainData, testData = one_hot("Embarked", trainData, testData)
    # 2.3 特征选择
    xTest = testData.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    yTrain = trainData["Survived"]
    xTrain = trainData.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    print(xTrain)
    print(yTrain)

    # 建立模型
    # 随机森林
    randomForest = RandomForestClassifier(n_estimators=100, random_state=42)
    randomForest.fit(xTrain, yTrain)
    yPred = randomForest.predict(xTest)
    save("./result/submission_11.csv", testData, yPred)
