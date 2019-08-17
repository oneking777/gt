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
    input_queue = tf.train.string_input_producer(path_filename_list)

    # 2，csv文件阅读器，默认只读取一个样本
    reader = tf.TextLineReader()

    key, value = reader.read(input_queue)

    # 3, csv文件解码器
    # record_defaults:指定每一个样本的每一列的类型，指定默认值[["None"], [4.0]]
    records = [['None'], ['None']]

    example, label = tf.decode_csv(value, record_defaults=records)

    # 4,样本批量处理
    example_batch, label_batch = tf.train.batch([example, label], batch_size=9, num_threads=1, capacity=9)

    return example_batch, label_batch


def imagereader(path_filename_list):
    """

    :param path_filename_list:
    :return: 指定个数的张量
    """
    # 1,文件队列的构造
    input_queue = tf.train.string_input_producer(path_filename_list)

    # 2,图片文件的读取
    reader = tf.WholeFileReader()

    key, value = reader.read(input_queue)

    # 3,图片文件进行解码
    image = tf.image.decode_jpeg(value)

    # 4,统一图片的形状
    image_resize = tf.image.resize_images(image, [200, 200])

    # 5,注意：一定要把样本的形状固定 [200, 200, 3],在批处理的时候要求所有数据形状必须定义
    image_resize.set_shape([200, 200, 3])

    # 6,批处理
    image_bath = tf.train.batch([image_resize], batch_size=20, num_threads=1, capacity=20)

    return image_bath


class CifarReader(object):
    """完成二进制文件的读取，写入tfrecords，并从tfrecords读取数据
    """
    def __init__(self, filelist):
        self.filelist = filelist
        # 定义图片属性
        self.high = 32
        self.width = 32
        self.channel = 3
        # 定义特征值和标签值
        self.label_bytes = 1
        self.image_bytes = self.high * self.width * self.channel
        self.bytes = self.label_bytes + self.image_bytes

    def read_and_decode(self):
        # 1,构造文件队列
        file_queue = tf.train.string_input_producer(self.filelist)

        # 2，二进制文件读取器
        reader = tf.FixedLengthRecordReader(record_bytes=self.bytes)

        key, value = reader.read(file_queue)

        # 3,二进制文件解码器
        label_image = tf.decode_raw(value, tf.uint8)

        # 4,分割标签值和特征值
        label = tf.cast(tf.slice(label_image, [0], [self.label_bytes]), tf.int32)
        image = tf.slice(label_image, [1], [self.image_bytes])  # uint8
        print(image)

        # 5,对图片特征值形状进行改变  [32*32*1]--->[32, 32, 1]
        image_reshape = tf.reshape(image, [self.high, self.width, self.channel])

        # 6,批量处理数据
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)

        return image_batch, label_batch

    def write_to_tfrecords(self, label_batch, image_batch):
        """
        将读出的二进制文件写入tfrecords文件中
        :param label_batch:
        :param image_batch:
        :return:None
        """
        # 1、建立TFRecord存储器
        writer = tf.python_io.TFRecordWriter('./tfrecords_files/test.tfrecords')

        for i in range(10):
            # 2，将样本写入文件中，图片样本构造example协议
            image = image_batch[i].eval().tostring()

            print(label_batch[i].eval()[0])
            label = int(label_batch[i].eval()[0])

            # 构造一个样本的example
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))

            # 写入单独的样本
            writer.write(example.SerializeToString())

        # 关闭
        writer.close()
        return None

    def read_from_records(self):
        """
        从tfrecords文件中读取数据
        :return:
        """
        # 1,构造文件队列
        file_queue = tf.train.string_input_producer(['./tfrecords_files/test.tfrecords', ])

        # 2，创建文件阅读器
        reader = tf.TFRecordReader()

        key, value = reader.read(file_queue)

        # 3、解析example
        features = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
        })

        # 4、解码内容, 如果读取的内容格式是string需要解码， 如果是int64,float32不需要解码
        image = tf.decode_raw(features["image"], tf.uint8)

        # 5,固定图片的形状，方便批处理
        image_reshape = tf.reshape(image, [self.high, self.width, self.channel])

        label = tf.cast(features["label"], tf.int32)

        # 进行批处理
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)

        return image_batch, label_batch






if __name__ == "__main__":
    # # 获取csv文件的名字的列表
    # filename_list = os.listdir("./image/")
    #
    # # 获取path+csv的列表
    # path_filename_list = [os.path.join("./image/", filename) for filename in filename_list]
    #
    # image_bath = imagereader(path_filename_list)
    #
    # # 开启会话
    # with tf.compat.v1.Session() as sess:
    #     # 线程协调器
    #     coord = tf.train.Coordinator()
    #
    #     # 开启线程操作
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #
    #     print(sess.run(image_bath))
    #
    #     # 回收子线程
    #     coord.request_stop()
    #
    #     coord.join(threads)
    # 获取path+name 的列表
    file_list = os.listdir("./image/")

    path_file_list = [os.path.join("./image/", file) for file in file_list]

    reader_01 = CifarReader(path_file_list)
    image_batch, label_batch = reader_01.read_and_decode()

    # 开启会话
    with tf.Session() as sess:
        # 开启线程协调器
        coord = tf.train.Coordinator()

        # 开启线程操作
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 写入tfrecords文件
        image_batch, label_batch = reader_01.read_from_records()
        print(sess.run([image_batch, label_batch]))

        # 回收线程
        coord.request_stop()
        coord.join(threads)


