import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error


def kc_house():
    """
    回归算法预测房价(梯度下降， 岭回归), 均方误差检验
    :return:None
    """
    # 获取数据
    data = pd.read_csv("./kc_house_data.csv")

    # 区分特征值， 目标值
    # 列索引
    # ['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',\
    #  'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement'\, 'yr_built', \
    # 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
    columns = list(data.columns)

    # 特征值['bedrooms', ...]
    data_x = data[columns[3:]]
    # 目标值['price']
    data_y = data[columns[2]]

    # 分割测试集和训练集
    x_train, x_test, y_train, y_test_font = train_test_split(data_x, data_y, test_size=0.25)

    # 特征值标准化
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值标准化
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.values.reshape(-1, 1))

    # 梯度下降对房价进行预测
    sgd = SGDRegressor()

    sgd.fit(x_train, y_train)

    y_predict = std_y.inverse_transform(sgd.predict(x_test))

    print("梯度下降预测的房价为：", y_predict)
    print("梯度下降均方误差为：", mean_squared_error(y_test_font, y_predict))

    # 岭回归对房价进行预测
    rg = Ridge()

    param = {"alpha": [0.5, 1.0, 3.0, 7.0, 10.0, 20.0, 40.0, 100.0]}
    gc = GridSearchCV(rg, param_grid=param, cv=10)

    gc.fit(x_train, y_train)

    y_predict = std_y.inverse_transform(gc.predict(x_test))

    print("岭回归预测的房价为：", y_predict)
    print("岭回归均方误差为：", mean_squared_error(y_test_font, y_predict))
    print("选取的最好的模型为：", gc.best_params_)

    return None


if __name__ == "__main__":
    kc_house()