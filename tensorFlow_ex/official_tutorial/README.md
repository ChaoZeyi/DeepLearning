##### one_hot

在读取数据集时经常会用到的one_hot属性(one_hot=True)，经常用于多标签数据集，表示一种二进制编码方式

例如在MNIST中，总共有0-9,10种标签，通过one_hot可表示为：

0 = 1000000000

1 = 0100000000

5 = 0000010000

非0(即为1)元素的索引代表的就是类别

https://www.quora.com/What-does-the-one_hot-True-parameter-on-the-MNIST-tensorflow-for-beginners-example-mean