import warnings

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
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
    # 提取Cabin的甲板号，将NaN视为 "U"（代表Unknown）
    trainData["Deck"] = trainData["Cabin"].apply(lambda x: x[0] if pd.notna(x) else "U")
    testData["Deck"] = testData["Cabin"].apply(lambda x: x[0] if pd.notna(x) else "U")


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


"""
1. 数据预处理
2. 特征选择
3. 数据划分
4. 模型训练
5. 模型评估
6. 参数调优
"""
if __name__ == "__main__":
    trainData = pd.read_csv("./titanic/train.csv")
    testData = pd.read_csv("./titanic/test.csv")
    # step1: 数据分析以及预处理
    # dataPreview()  # 数据查看
    # analysisDataByPicture()  # 数据分析（通过画图）

    fillByAverage("Age")  # 对年龄填充平均值
    fillByMost("Embarked")  # 登船港口，有两个缺失值，数量较少，用出现频率最多的填充即可
    fillByUnknown("Cabin")  # 填充夹板号，有大量缺失值，通常不会尝试去预测或估计具体的船舱号，也不能删除这个数据，因为其他的字段是有用的，这里我们把它表示为unknown
    fillByAverage("Fare")  # 填充票价，平均值

    # step2: 特征选择和特征工程
    # 2.1 sex: 转换为0-1，embarked: 使用one-hot编码
    classifierByZero_One("Sex", "male", "female")
    trainData, testData = one_hot("Embarked", trainData, testData)

    # 2.2 创建新的特征
    dropCols = []
    # 从Name中提取称谓作为新特征Title
    trainData["Title"] = trainData["Name"].apply(lambda name: name.split(",")[1].split(".")[0].strip())
    testData["Title"] = testData["Name"].apply(lambda name: name.split(",")[1].split(".")[0].strip())
    # Title
    grid = sns.FacetGrid(trainData, col='Title')
    grid.map(sns.pointplot, 'Title', palette='deep')
    grid.add_legend()

    # plt.show()

    dropCols.append("Name")
    # Deck 是从 Cabin 的第一个字母提取的，代表乘客的船舱甲板位置，这可能会影响他们在紧急情况下的逃生机会。
    dropCols.append("Cabin")
    # TicketPrefix 提取自 Ticket，可能包含了与票价或船舱位置相关的信息.
    trainData["TicketPrefix"] = trainData["Ticket"].apply(lambda x: x.split()[0] if not x.isdigit() else "None")
    testData["TicketPrefix"] = testData["Ticket"].apply(lambda x: x.split()[0] if not x.isdigit() else "None")
    dropCols.append("Ticket")
    # 查看数据集中新构造的特征
    # print(trainData[["Title", "FamilySize", "IsAlone", "Deck", "TicketPrefix"]].head())

    # 2.3 特征选择
    testData = pd.get_dummies(testData, columns=["Deck", "TicketPrefix","Title"])
    xTest = testData.drop(dropCols, axis=1)
    dropCols.append("Survived")
    yTrain = trainData["Survived"]
    trainData = pd.get_dummies(trainData, columns=["Deck", "TicketPrefix","Title"])
    xTrain = trainData.drop(dropCols, axis=1)
    # print(xTrain)
    # 利用随机森林分类器，在训练数据上利用SelectFromModel根据随机森林估计出的特征重要性来选择特征。
    # 选择标准 threshold='mean' 意味着将选择重要性大于平均重要性的特征。
    randomForest = RandomForestClassifier(n_estimators=100, random_state=42)

    randomForest.fit(xTrain, yTrain)
    selector = SelectFromModel(randomForest, threshold="mean", prefit=True)
    xTrainWithImportantFactor = selector.transform(xTrain)
    importantFeatureName = xTrain.columns[selector.get_support()]
    print(importantFeatureName)
    importantFeatureNameList = []
    # importantFeatureNameList2 = ["PassengerId"]
    for idx in range(0, len(importantFeatureName)):

        importantFeatureNameList.append(importantFeatureName[idx])
    # 删除不再需要的原始列

    trainData = xTrain[importantFeatureNameList]

    testData=xTest[importantFeatureNameList]
    # importantFeatureNameList2.extend(importantFeatureNameList)
    print(trainData)
    print(testData)
    # testData2=testData
    # step3: 数据划分
    # 此处的数据划分是针对训练集，在训练集的返回内再次划分为训练集和测试集，为了避免数据不够，应将test_size设置的小一些

    xTrain, xTest, yTrain, yTest = train_test_split(trainData, yTrain, test_size=0.1, random_state=42)

    # step4: 建立模型
    # model1: 随机森林
    # 模型初始化
    model_1 = RandomForestClassifier(n_estimators=100, random_state=42)

    # 模型训练
    model_1.fit(xTrain, yTrain)
    # 模型预测
    yPred = model_1.predict(xTest)
    # 模型评估
    print("准确率:", accuracy_score(yTest, yPred))
    print("分类结果:\n", classification_report(yTest, yPred))

    # step5_6: 模型评估以及参数调优
    param_grid = {
        'n_estimators': [100, 150, 200]
    }
    # 采用网格搜索
    grid_search = GridSearchCV(estimator=model_1, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(xTrain, yTrain)

    # 最佳参数模型
    best_model_1 = grid_search.best_estimator_

    # 模型评估
    yPred = best_model_1.predict(xTest)
    print("参数调优后的准确率:", accuracy_score(yTest, yPred))
    print("参数调优后的分类结果:\n", classification_report(yTest, yPred))

    # # 正式测试
    # yPred = best_model_1.predict(testData)

    #
    # # 数据提交
    # # 生成提交文件
    # count = 1
    # save("./result/submission_"+ str(count) + ".csv", testData, yPred)
    # count += 1
    #
    # model2: SVC
    # svc = SVC()
    # svc.fit(xTrain, yTrain)
    # yPred = svc.predict(testData)
    #
    # save("./result/submission_"+ str(count) + ".csv", testData, yPred)
    # count += 1
    #
    # # model3: 逻辑回归
    # logReg = LogisticRegression(random_state=42)
    # logReg.fit(xTrain, yTrain)
    # yPred = logReg.predict(testData)
    # save("./result/submission_"+ str(count) + ".csv", testData, yPred)
    # count += 1
    #
    # # model4: 高斯-朴素贝叶斯
    # gaussian = GaussianNB()
    # gaussian.fit(xTrain, yTrain)
    # yPred = gaussian.predict(testData)
    # save("./result/submission_"+ str(count) + ".csv", testData, yPred)
    # count += 1

    # model5: 梯度提升
    gBC = GradientBoostingClassifier(random_state=42)
    gBC.fit(xTrain, yTrain)
    yPred = gBC.predict(testData)
    scores = cross_val_score(gBC, xTrain, yTrain, cv=5)  # cv=5表示使用5折交叉验证
    print("交叉验证准确率:", scores.mean())  # 输出平均准确率
    save("./result/submission_123.csv", testData, yPred)
    # count += 1

    # # model6: k近邻
    # knn = KNeighborsClassifier()
    # knn.fit(xTrain, yTrain)
    # yPred = knn.predict(testData)
    # save("./result/submission_" + str(count) + ".csv", testData, yPred)
    # count += 1
    #
    # # model7: 决策树
    # decision_tree = DecisionTreeClassifier()
    #
    # decision_tree.fit(xTrain, yTrain)
    # yPred = decision_tree.predict(testData)
    # save("./result/submission_" + str(count) + ".csv", testData, yPred)
    # count += 1
