import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = 'FALSE'

# a = tf.constant(3.0)
# b = tf.constant(5.0)
#
# sum1 = tf.add(a, b)
#
# input_1 = tf.placeholder(tf.float32)
# input_2 = tf.placeholder(tf.float32)
#
# output = tf.add(input_1, input_2)
#
# c = np.arange(24).reshape(4, 6)
#
# with tf.Session() as sess:
# print(sess.run(sum1))
# print(a.graph)
# print(sess.run([output], feed_dict = {input_1: 10.0, input_2: 12.0}))
# print(output.op)
# print(output.name)
# print(output.shape)
# print(c.shape)

# 形状的概念
# 静态形状和动态形状
# 对于静态形状而言，一旦张良形状固定了，则不能再次设置静态形状
# plt = tf.compat.v1.placeholder(tf.float32, [None, 2])
#
# print(plt)
#
# plt.set_shape([3, 2])
#
# print(plt)
#
# plt_reshape = tf.reshape(plt, [2, 3])
#
# print(plt_reshape)
#
# with tf.compat.v1.Session() as sess:
#     pass


# 变量的创建
# 1，变量能够持久化保存，普通张量op是不行的
# 2，当定义一个变量op时，一定要在会话当中运行初始化
# 3,name参数：在tensorboard使用的时候显示名字，可以让相同op名字进行区分
#
# a = tf.constant(3.0)
# b = tf.constant(4.0)
#
# c = tf.add(a, b)
#
# var = tf.Variable(tf.random.normal([2, 3], mean=0.0, stddev=1.0))
#
# print(a, var)
#
# # 必须做一步显示的初始化
# init_op = tf.compat.v1.global_variables_initializer()
#
# with tf.compat.v1.Session() as sess:
#     # 必须运行初始化op
#     sess.run(init_op)
#
#     # 把程序的图结构写入事件, graph: 把指定的图写进事件文件当中
#     filewriter = tf.compat.v1.summary.FileWriter("./summary/test_01/", graph=sess.graph)
#
#     print(sess.run([c, var]))

# 训练参数问题：trainable
# 学习率和步数的设置

# 添加权重参数，损失值等在tensorboard观察情况 1,收集变量 2，合并变量写入事件文件


def mylineregression():
    """
    自实现一个线性回归预测
    :return: None
    """
    with tf.compat.v1.variable_scope("x_data"):
        # 1,准备数据，x：特征值[100, 10]   y:目标值[100]
        x = tf.random.normal([100, 1], mean=1.75, stddev=0.5, name="x_data")

        # 矩阵相乘必须是二维
        y_true = tf.compat.v1.sparse_matmul(x, [[0.7]]) + 0.8

    with tf.compat.v1.variable_scope("model"):
        # 2,建立线性回归模型, 1个特征，1个权重，1个偏置， y = x w + b
        # 随机给一个权重和偏置的值， 当前状态下优化
        # trainable参数：指定这个变量能跟着梯度下降一起优化
        weight = tf.Variable(tf.random.normal([1, 1], mean=0.0, stddev=1.0), name="w")
        bias = tf.Variable(0.0, name="b")

        y_predict = tf.compat.v1.sparse_matmul(x, weight) + bias

    with tf.compat.v1.variable_scope("loss"):
        # 3,建立损失函数，均方误差
        loss = tf.compat.v1.reduce_mean(tf.square(y_true - y_predict))

    with tf.compat.v1.variable_scope("optimizer"):
        # 4， 梯度下降优化损失 learning_rating: 0~1, 2, 5, 7, 10
        train_op = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.compat.v1.variable_scope("init_op"):
        # 定义初始化变量的op
        init_op = tf.compat.v1.global_variables_initializer()

    # 1,收集tensor
    tf.compat.v1.summary.scalar("losses", loss)
    tf.compat.v1.summary.histogram("weights", weight)

    # 定义合并tensor的op
    merged = tf.compat.v1.summary.merge_all()

    # 定义一个保存模型的op
    saver = tf.compat.v1.train.Saver()

    # 通过会话运行程序
    with tf.compat.v1.Session() as sess:

        # 初始化变量
        sess.run(init_op)

        # 打印随机初始化的权重
        print("随机初始化的权重为：{}, 偏置为{}".format(weight.eval(), bias.eval()))

        # 建立事件文件
        filewriter = tf.compat.v1.summary.FileWriter("./summary/test_01/", graph=sess.graph)

        # 加载模型，覆盖模型当中随机定义的参数，从上次训练的参数结果开始
        if os.path.exists("./ckpt/checkpoint"):
            saver.restore(sess, "./ckpt/model")

        # 循环训练，运行优化
        for i in range(500):
            sess.run(train_op)

            # 运行合并的tensor
            summary = sess.run(merged)

            filewriter.add_summary(summary, i)

            print("第{}次优化参数的权重为：{}, 偏置为{}".format(i, weight.eval(), bias.eval()))

        saver.save(sess, "./ckpt/model")

    return None


if __name__ == "__main__":
    mylineregression()