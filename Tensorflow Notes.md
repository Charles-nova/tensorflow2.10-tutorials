## Tensorflow安装

可以直接使用pip进行安装，其中比较麻烦一点的是cuda比较新的版本安装。

`conda install cudatoolkit=11.2 cudnn=8.1.0`找不到包，最后采用了直接去镜像源下载`tar.bz2`安装包，使用`conda install --use-local xxxx`

## jupyter notebook安装

`pip install jupyter` `pip install ipykerner`

添加虚拟环境到notebook `python -m ipykernel install --name=web_env(注册名字)`

添加插件

`pip install jupyter_contrib_nbextensions`

`jupyter contrib nbextension install`

`pip install jupyter_nbextensions_configurator`

`jupyter nbextensions_configurator enable --user`

## Tensorflow自定义层

1. 继承tf.keras.layers.Layer类

2. init() 负责初始化对象，给定层的输出维数。  

3. build() 为层增加初始化可变权重和可变偏置量。  

4. call() 则是按层的运算公式返回值

### TensorFlow 中的函数

1. `tf.control_dependencies()`返回一个控制依赖的上下文管理器, 指定某些操作执行的依赖关系.

2. `tf.while_loop(cond, body, loop_vars)` tf中的循环函数

## Tensorflow 2.X执行模式

1. Eager Execution：它是一个命令式、由运行定义的接口，一旦从 Python 被调用，其操作立即被执行。
   
   Eager Execution 的优点如下：
   
   - 快速调试即刻的运行错误并通过 Python 工具进行整合
   - 借助易于使用的 Python 控制流支持动态模型
   - 为自定义和高阶梯度提供强大支持
   - 适用于几乎所有可用的 TensorFlow 运算

2. 预先定义计算图，运行时反复使用，不能改变；速度更快，适合大规模部署，适合嵌入式平台。目前推荐使用@tf.function装饰器来进行计算图的构建。控制流使用tf原生的函数比如`tf.cond(), tf.while_loop(), tf.add(), tf.not_equal()`
   
   当被 @tf.function 修饰的函数第一次被调用的时候，进行以下操作：
   
   **（1）** 在 Eager Execution 模式关闭的环境下，函数内的代码依次运行。也就是说，每个 tf. 方法都只是定义了计算节点，而并没有进行任何实质的计算。这与 TensorFlow 1.X 的 Graph Execution 是一致的；
   
   **（2）** 使用 AutoGraph 将函数中的 Python 控制流语句转换成 TensorFlow 计算图中的对应节点（比如说 while 和 for 语句转换为 tf.while ， if 语句转换为 tf.cond 等等；
   
   **（3）** 基于上面的两步，建立函数内代码的计算图表示；然后运行一次这个计算图；
   
   **（4）** 基于函数的名字和输入的函数参数的类型生成一个哈希值，并将建立的计算图缓存到一个哈希表中。如果在被 @tf.function 修饰的函数之后再次被调用的时候，根据函数名和输入的函数参数的类型计算哈希值，检查哈希表中是否已经有了对应计算图的缓存。如果是，则直接使用已缓存的计算图，否则重新按上述步骤建立计算图。
   
   ## Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice
   
   已经解决了：[找不到TensorFlow libdevice。为什么在搜索的路径中找不到它？ - 问答 - Python中文网](https://www.cnpython.com/qa/1471441) 要将nvvm这个文件夹(文件来源是直接windows安装的cuda11.3)移入到apath所指的文件夹中。
   
   ## Call to CreateProcess failed. Error code: 2 (TensorFlow)
   
   安装nvidia cuda-nvcc
   
   # os.environ
   
   该模块获取环境变量。环境变量是程序和操作系统之间的通信方式。有些字符不宜明文写进代码里，比如数据库密码，个人账户密码，如果写进自己本机的环境变量里，程序用的时候通过 os.environ.get() 取出来就行了。
   
   XLA is an optimizing graph compiler for TensorFlow. It optimizes parts of the TensorFlow GraphDef in an attempt to improve performance.
   
   Unlike native TensorFlow, which executes GraphDef nodes one at a time, XLA considers many GraphDef nodes at once and generates optimized code for these nodes

# keras functional API

The functional API can handle models with non-linear topology, shared layers, and even multiple inputs or outputs.

The main idea is that a deep learning model is usually a directed acyclic graph (DAG) of layers. So the functional API is a way to build *graphs of layers*

`kears.Input`  and `keras.Model`

`tf.keras.layers.Normalization`在使用前必须调用`adapt()`方法，adapt()方法的注意事项在 https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/PreprocessingLayer#adapt
