import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = 'FALSE'


# 模拟同步先处理数据，然后才取训练数据

# 1，首先定义队列
# Q = tf.queue.FIFOQueue(3, tf.float32)
#
# # 放入一些数据
# enq_many = Q.enqueue_many([[0.1, 0.2, 0.3], ])
#
# # 2，定义一些读取数据，取数据的过程     取数据，+1，入队列
#
# out_q = Q.dequeue()
#
# data = out_q + 1
#
# en_q = Q.enqueue(data)
#
#
# with tf.compat.v1.Session() as sess:
#     # 初始化队列
#     sess.run(enq_many)
#
#     # 处理数据
#     for i in range(100):
#         sess.run(en_q)
#
#     # 训练数据
#     for i in range(Q.size().eval()):
#         print(sess.run(Q.dequeue()))


# 模拟异步子线程存入样本，主线程读取样本

# 1，定义一个队列，1000
# Q = tf.queue.FIFOQueue(1000, tf.float32)
#
# # 2，定义子线程要做的事情   值，+1，  放入队列
# var = tf.Variable(0.1)
#
# # 实现自增，tf.assign_add
# data = tf.compat.v1.assign_add(var, tf.constant(1.0))
#
# en_q = Q.enqueue(data)
#
# # 3，定义队列管理器op，指定子线程要做的事情
# qr = tf.compat.v1.train.queue_runner.QueueRunner(Q, enqueue_ops=[en_q] * 2)
#
# # 初始化变量op
# init_op = tf.compat.v1.global_variables_initializer()
#
# with tf.compat.v1.Session() as sess:
#     # 初始化变量
#     sess.run(init_op)
#
#     # 设置线程协调器
#     coord = tf.train.Coordinator()
#
#     # 开启子线程
#     threads = qr.create_threads(sess, coord=coord, start=True)
#
#     # 主线程，不断读取数据训练
#     for i in range(300):
#         sess.run(Q.dequeue())
#
#     # 停止子线程
#     coord.request_stop()
#
#     # 回收线程资源
#     coord.join(threads)


# csv文件读取案例

def csvreader(path_filename_list):
    """
    :param:包含路径的文件名字的列表
    :return:指定个数的张量
    """
    # 1,文件队列的构造
    input_queue = tf.compat.v1.train.string_input_producer(path_filename_list)

    # 2，csv文件阅读器，默认只读取一个样本
    reader = tf.compat.v1.TextLineReader()

    key, value = reader.read(input_queue)

    # 3, csv文件解码器
    # record_defaults:指定每一个样本的每一列的类型，指定默认值[["None"], [4.0]]
    records = [['None'], ['None']]

    example, label = tf.io.decode_csv(records=value, record_defaults=records)

    # 4,样本批量处理
    example_batch, label_batch = tf.compat.v1.train.batch([example, label], batch_size=9, num_threads=1, capacity=9)

    return example_batch, label_batch


if __name__ == "__main__":
    # 获取csv文件的名字的列表
    filename_list = os.listdir("./data_csv/")

    # 获取path+csv的列表
    path_filename_list = [os.path.join("./data_csv/", filename) for filename in filename_list]

    example_batch, label_batch = csvreader(path_filename_list)

    # 开启会话
    with tf.compat.v1.Session() as sess:
        # 线程协调器
        coord = coord = tf.train.Coordinator()

        # 开启线程操作
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        print(sess.run(example_batch))
        print("*"*50)
        print(sess.run(label_batch))

        # 回收子线程
        coord.request_stop()

        coord.join(threads)

