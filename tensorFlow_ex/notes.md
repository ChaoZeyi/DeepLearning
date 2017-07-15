### 基础语法

#### 数据类型

a = tf.constant(1, dtype=tf.float32)	常量

b = tf.Variable(1, dtype=tf.float32)	变量

c = tf.placeholder(dtype=tf.float32)	占位符

变量初始化（必须初始化后才可以使用）：

init = tf.global_variables_initializer()
sess.run(init)

变量重赋值：

sess.run(tf.assign(b, 2))

在能够自动训练机器学习模型的Optimizer类，这意味着这些类将自动修改Variable对象的值，而无须显式做出请求。但如果要求一些Variable对象只可手工修改，而不允许使用Optimizer类时，可在创建这些Variable对象时将其trainable参数设
为False：b = tf.Variable(1, dtype=tf.float32, trainable=False)

标量：0阶张量

向量：1阶张量

矩阵：N阶张量

#### 数据操作

|      |         加         |            减            |            乘            |           除           |
| :--: | :---------------: | :---------------------: | :---------------------: | :-------------------: |
|  数字  | tf.add(a,b)   a+b | tf.subtract(a,b)    a-b | tf.multiply(a,b)    a*b | tf.divide(a,b)    a/b |
|  矩阵  | tf.add(a,b)   a+b | tf.subtract(a,b)    a-b |     tf.matmul(a,b)      |                       |

**在对矩阵操作时，同样可以使用tf.multiply(a,b)或a*b，但这两种方式得到了不是矩阵乘法，而是矩阵元素对应相乘**

![1499865929(1).jpg](https://github.com/ChaoZeyi/DeepLearning/blob/master/tensorFlow_ex/official_tutorial/photos/1499865929(1).jpg?raw=true)

**tf.matmul(a,b)与tf.matmul(b,a)不相等**

tf.divide(a,b)和a/b也是同理

求逆矩阵：c = tf.matrix_inverse(a)

reduce_sum(a)，求张量各维数的和。a = [[1,2],[3,4]]，reduce_sum(a) = 10

### FAQ

##### 基本步骤

将模型参数作为变量，数据输入作为占位符，根据模型结构写出输出与输入、参数之间的关系式，得到代价函数，选择优化算法进行训练，同时确定训练的学习速率和训练次数。初始化变量，开始训练，然后测试训练效果。

##### tensorBoard

tf.summary.FileWriter('路径/myGraph', sess.graph)

进入该目录下，运行>tensorboard --logdir="myGraph"

运行之后，相当于开启了tensorFlow，浏览http://localhost:6006

##### one_hot

在读取数据集时经常会用到的one_hot属性(one_hot=True)，经常用于多标签数据集，表示一种二进制编码方式

例如在MNIST中，总共有0-9,10种标签，通过one_hot可表示为：

0 = 1000000000

1 = 0100000000

5 = 0000010000

非0(即为1)元素的索引代表的就是类别

https://www.quora.com/What-does-the-one_hot-True-parameter-on-the-MNIST-tensorflow-for-beginners-example-mean

##### graph

> 加载tensorflow库时，会自动创建一个graph对象，作为默认的数据流图
>
> 使用g1 = tf.get_default_graph()获取默认数据流图的句柄
>
> g2 = tf.Graph()创建新的数据流图
>
> with g1.as_default():
>
> ​	Op
>
> with g2.as_default():
>
> ​	Op
>
> 分别为两个数据流图添加Op，如果没有指定数据流图，则加入默认数据流图

**在大多数TensorFlow程序中，只使用默认数据流图就足够了。然而，如果需要定义多个相互之间不存在依赖关系的模型，则创建多个Graph对象十分有用。**

如果存在多个数据流图，在调用Session()时，需要显示指定运行的是哪个graph，不然会默认执行默认数据流图

##### 检查点（checkpoint)

tf.train.Saver

