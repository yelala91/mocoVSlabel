# MoCo算法和监督学习算法对比

本实验实现了 MoCo 自监督学习算法，并和监督学习算法进行了一定的对比。

## 数据集准备

下载 TinyImageNet 数据集，下面我们记该数据集的完整地址为 {tinyimagenet}

```
http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

下载 CIFAR-100 数据集，下面我们记该数据集的完整地址为 {cifar-100}

```
https://www.cs.toronto.edu/~kriz/cifar.html
```



## 训练

### 训练 MoCo 

在终端输入如下命令，其中要求数据集的地址是完整地址，不是相对地址。

```
python main.py {tinyimagenet} \
                --mode train  \
                --model moco  \
                -o {输出地址}  \
                --lr {学习率}  \
                -b {batch size}\
                -e {epochs}   \
                -j {number workers} \ 
                ... 其他可选参数可见文件 utils.py
```

### 冻结隐藏层训练最后一个全连接层

在终端输入如下命令，其中要求数据集的地址是完整地址，不是相对地址。

```
python main.py {cifar-100} \
            --mode LCPtrain \
            ---model moco  \
            -o {输出地址}   \
            --weight {权重地址} \
            --lr {学习率}  \
            -b {batch size}\
            -e {epochs}   \
            -j {number workers} \ 
            ... 其他可选参数可见文件 utils.py
```

### 微调模型

在终端输入如下命令，其中要求数据集的地址是完整地址，不是相对地址。

```
python main.py {cifar-100} \
            --mode ft      \
            ---model {需要微调的模型} \  moco 或 resnet 
            -o {输出地址}   \
            --weight {权重地址} \
            --lr {学习率}  \
            -b {batch size}\
            -e {epochs}   \
            -j {number workers} \ 
            ... 其他可选参数可见文件 utils.py
```

## 测试

在终端输入如下命令，其中要求数据集的地址是完整地址，不是相对地址。

```
python main.py {cifar-100} \
            --mode test \
            ---model {模型} \  moco 或 resnet 
            -o {输出地址}   \
            --weight {权重地址} \
            --lr {学习率}  \
            -b {batch size}\
            -e {epochs}   \
            -j {number workers} \ 
            ... 其他可选参数可见文件 utils.py
```