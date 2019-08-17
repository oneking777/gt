from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor, Ridge
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


def iris():
    """
    knn预测鸢尾花类别
    :return: None
    """
    # 获取数据
    ir = load_iris()

    # 分割测试集和训练集数据
    x_train, x_test, y_train, y_test = train_test_split(ir.data, ir.target)

    # 对训练数据进行标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # knn算法进行预测
    knn = KNeighborsClassifier()

    param = {"n_neighbors": [3, 5, 8, 12, 16]}

    # 网格搜索及交叉验证
    gc = GridSearchCV(knn, param_grid=param, cv=10)

    gc.fit(x_train, y_train)

    y_predict = gc.predict(x_test)

    print("knn预测测试集数据值为：", y_predict)
    print("knn算法预测的准确率为：", gc.score(x_test, y_test))
    print("准确率及召回率为：", classification_report(y_test, y_predict, target_names=['长', '高', '宽'], labels=[0, 1, 2]))
    print("最好的模型为：", gc.best_params_)
    return None


def news20():
    """
    朴素贝叶斯预测新闻分类
    :return: None
    """
    # 获取数据
    news = fetch_20newsgroups(subset='all')

    # 分割测试集训练集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)

    # 生成文章特征词
    # 实例化特征抽取类
    # cv = CountVectorizer()
    tf = TfidfVectorizer()
    # 调用fit_transform
    # x_train = cv.fit_transform(x_train)
    # x_test = cv.transform(x_test)
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)

    # 朴素贝叶斯进行预测
    mm = MultinomialNB(alpha=1.0)

    mm.fit(x_train, y_train)
    y_predict = mm.predict(x_test)

    print("朴素贝叶斯预测测试集结果为：", y_predict)
    print("朴素贝叶斯预测准确率为：", mm.score(x_test, y_test))

    return None


def taitan_tree():
    """
    决策树预测泰坦尼克号好存活概率
    :return: None
    """
    # 获取数据
    data = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

    # 特征值， 目标值
    data_x = data[["pclass", "age", "sex"]]
    data_y = data["survived"]

    # 填充年龄nan值
    data_x["age"].fillna(data_x["age"].mean(), inplace=True)

    # 分割测试集和训练集
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y)

    # 特征工程处理，dict特征抽取
    dv = DictVectorizer()

    x_train = dv.fit_transform(x_train.to_dict(orient="records"))
    x_test = dv.transform(x_test.to_dict(orient="records"))

    # 决策树进行预测
    dt = DecisionTreeClassifier()

    param = {"max_depth": [10, 12, 15, 18, 25]}
    gc = GridSearchCV(dt, param_grid=param, cv=10)

    gc.fit(x_train, y_train)

    y_predict = gc.predict(x_test)

    # print("决策树预测测试集结果为：", y_predict)
    print("准确率为：", gc.score(x_test, y_test))
    print("决策树预测的准确率和召回率为：", classification_report(y_test, y_predict, labels=[0, 1], target_names=['死亡', '存活']))
    print("选取的模型为：", gc.best_params_)

    return None


def taitan_forest():
    """
    随机森林预测泰坦尼克存活人数
    :return: None
    """
    # 获取数据
    data = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

    # 特征值， 目标值
    data_x = data[["pclass", "age", "sex"]]
    data_y = data["survived"]

    # 填充年龄nan值
    data_x["age"].fillna(data_x["age"].mean(), inplace=True)

    # 分割测试集和训练集
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y)

    # 特征工程处理，dict特征抽取
    dv = DictVectorizer()

    x_train = dv.fit_transform(x_train.to_dict(orient="records"))
    x_test = dv.transform(x_test.to_dict(orient="records"))

    # 随机森林进行预测存活
    rf = RandomForestClassifier()

    param = {"n_estimators": [10, 12, 15, 20], "max_depth": [10, 12, 15, 18, 25]}
    gc = GridSearchCV(rf, param_grid=param, cv=10)

    gc.fit(x_train, y_train)

    y_predict = gc.predict(x_test)
    print("随机森林预测测试集结果为：", y_predict)
    print("准确率为：", gc.score(x_test, y_test))
    print("准确率和召回率：", classification_report(y_test, y_predict, labels=[0, 1], target_names=['死亡', '存活']))
    print("选取的模型为：", gc.best_params_)

    return None


def taitan_erfen():
    """
    逻辑回归预测泰坦尼克号存活
    :return: None
    """
    # 获取数据
    data = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

    # 特征值， 目标值
    data_x = data[["pclass", "age", "sex"]]
    data_y = data["survived"]

    # 填充年龄nan值
    data_x["age"].fillna(data_x["age"].mean(), inplace=True)

    # 分割测试集和训练集
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y)

    # 特征工程处理，dict特征抽取
    dv = DictVectorizer()

    x_train = dv.fit_transform(x_train.to_dict(orient="records"))
    x_test = dv.transform(x_test.to_dict(orient="records"))

    # 逻辑回归预测存活
    lr = LogisticRegression()

    lr.fit(x_train, y_train)

    y_predict = lr.predict(x_test)

    print("逻辑回归预测测试集结果为：", y_predict)
    print("逻辑回归预测准确率为：", lr.score(x_test, y_test))
    print("权重为：", lr.coef_)

    return None


def line():
    """
    正规方程预测波士顿房价
    :return: None
    """
    # 获取数据
    boston = load_boston()

    # 分割测试集和训练集
    x_train, x_test, y_train, y_test_front = train_test_split(boston.data, boston.target)

    # 对数据进行标准化
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test_front.reshape(-1, 1))

    # 正规方程对波士顿房价进行预测
    lr = LinearRegression()

    lr.fit(x_train, y_train)

    y_predict = std_y.inverse_transform(lr.predict(x_test))

    # print("正规方程对测试集预测结果为：", y_predict)
    print("正规方程准确率为：", lr.score(x_test, y_test))
    print("正规方程均方误差为：", mean_squared_error(y_test_front, y_predict))
    print("正规方程权重为：", lr.coef_)

    return None


def tidu():
    """
    梯度下降对波士顿房价进行预测
    :return: None
    """
    # 获取数据
    boston = load_boston()

    # 分割测试集和训练集
    x_train, x_test, y_train, y_test_front = train_test_split(boston.data, boston.target)

    # 对数据进行标准化
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test_front.reshape(-1, 1))

    # 梯度下降对波士顿房价进行预测
    sgd = SGDRegressor()

    sgd.fit(x_train, y_train)

    y_predict = std_y.inverse_transform(sgd.predict(x_test))

    # print("梯度下降对测试集预测结果为：", y_predict)
    print("梯度下降准确率为：", sgd.score(x_test, y_test))
    print("梯度下降均方误差为：", mean_squared_error(y_test_front, y_predict))
    print("梯度下降权重为：", sgd.coef_)

    return None


def ling():
    """
    岭回归预测波士顿房价
    :return: None
    """
    # 获取数据
    boston = load_boston()

    # 分割测试集和训练集
    x_train, x_test, y_train, y_test_front = train_test_split(boston.data, boston.target)

    # 对数据进行标准化
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test_front.reshape(-1, 1))

    # 梯度下降对波士顿房价进行预测
    rd = Ridge()

    param = {"alpha": [0.5, 1.0, 3.0, 5.0, 10.0]}
    gc = GridSearchCV(rd, param_grid=param, cv=10)

    gc.fit(x_train, y_train)

    y_predict = std_y.inverse_transform(gc.predict(x_test))

    # print("梯度下降对测试集预测结果为：", y_predict)
    print("岭回归准确率为：", gc.score(x_test, y_test))
    print("岭回归均方误差为：", mean_squared_error(y_test_front, y_predict))
    print("岭回归选取的模型为：", gc.best_params_)

    return None


def breast_cancer():
    """
    逻辑回归预测乳腺癌是否患病
    :return: None
    """
    column = ['Sample code number','Clump Thickness', 'Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

    # 获取数据
    data = pd.read_csv("../breast-cancer-wisconsin.data", names=column)
    # 替换？，处理缺失值
    data.replace(to_replace='?', value=np.nan, inplace=True)
    data.dropna(inplace=True)

    # 分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]], data[column[10]])

    # 标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 逻辑回归进行预测
    lr = LogisticRegression()

    lr.fit(x_train, y_train)

    y_predict = lr.predict(x_test)

    print("逻辑回归预测准确率为：", lr.score(x_test, y_test))
    print("准确率和召回率为：", classification_report(y_test, y_predict))
    print("权重为：", lr.coef_)

    return None


if __name__ == '__main__':
    breast_cancer()