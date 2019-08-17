import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


def dbts():
    """
    预测印第安人是否患糖尿病
    :return:None
    """
    # 获取数据
    data = pd.read_csv("./diabetes.csv")

    # 获取特征值， 目标值
    data_x = data.iloc[:, 0:8]
    data_y = data.iloc[:, 8]

    # 分割测试集和训练集
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25)

    # 特征值进行标准化
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 逻辑回归预测是否患病
    lgr = LogisticRegression()

    lgr.fit(x_train, y_train)

    y_predict = lgr.predict(x_test)

    print("逻辑回归准确率为：", lgr.score(x_test, y_test))
    print("逻辑回归召回率为：", classification_report(y_test, y_predict))

    # 随机森林预测是否患病
    rf = RandomForestClassifier()

    param = {"n_estimators": [3, 6, 10, 18, 30], "max_depth": [5, 8, 10, 15]}
    gc = GridSearchCV(rf, param, cv=10)

    gc.fit(x_train, y_train)

    y_predict = gc.predict(x_test)

    print("随机森林准确率为：", gc.score(x_test, y_test))
    print("随机森林召回率为：", classification_report(y_test, y_predict))
    print("随机森林最佳模型为：", gc.best_params_)

    return None


if __name__ == "__main__":
    dbts()