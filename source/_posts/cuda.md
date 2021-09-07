---
title: cuda 安装教程
date: 2021-07-19 23:11:09
categories: python
tags:
typora-root-url: cuda
---


一直是个电脑配置盲，显卡啥的都不太懂，~~大概学了个假cs~~。正在学习ML，DL，要用到pytorch, tenserflow啥的，需要用到 Nvidia cuda，正好安装一下，也捋一捋配置。



---

# 我的电脑配置

- 电脑型号：HUAWEI xpro
- 系统：Ubuntu 20.04.2 LTS 
  - 主要工作系统是 Linux，所以本篇基本是 Ubuntu 上的 Cuda 安装教程。
- 处理器：Intel® Core™ i5-8265U CPU @ 1.60GHz × 8 

<!--more -->

## 显卡

最主要就是看我们的显卡了，首先扫扫盲看看显卡是个什么东西。

参考[一篇知乎回答](https://www.zhihu.com/question/28422454)。

- **显卡（Vedio Card , Graphics Card）**，也叫显示适配器，主机里的数据要显示在屏幕上就需要显卡。因此，显卡是电脑进行数模信号转换的设备，承担输出显示图形的任务。具体来说，**显卡接在电脑主板上，它将电脑的数字信号转换成模拟信号让显示器显示出来**。
  - 集成显卡，是指显卡集成在主板上，不能随意更换。而独立显卡是作为一个独立的器件插在主板的AGP接口上的，可以随时更换升级。集成显卡使用物理内存，而独立显卡有自己的显存。
- **VGA(Video Graphics Array)**，是视频图形阵列，就是显示输出接口，能够输出一个分辨率下的色彩rgb数据。
- **GPU(Graphics Processing Unit)**，这个就是显卡上的一块芯片，就像 CPU 是主板上的一块芯片。关于GPU的作用。。。这个其实ics和os里都讲过(，指路[jyy的slides](http://jyywiki.cn/OS/2021/slides/13.slides#/4)。就是一个用于画图的硬件，进行大量的显存操作的像素计算（矩阵运算），能够加速图形渲染显示。当人们发现它的强大算力之后就有了。。。
- **CUDA(Compute Unified Device Architecture)**，通用并行计算架构，是一种运算平台。它包含CUDA指令集架构以及GPU内部的并行计算引擎。你只要使用一种类似于C语言的**CUDA C语言**，就可以开发CUDA程序，从而可以更加方便的利用GPU强大的计算能力，而不是像以前那样先将计算任务包装成图形渲染任务，再交由GPU处理。（炼丹）
- **cuDNN**，NVIDIA cuDNN是用于深度神经网络的GPU加速库。它强调性能、易用性和低内存开销。NVIDIA cuDNN可以集成到更高级别的机器学习框架中，如谷歌的Tensorflow。



大概就这些了，理清楚显卡是干啥的，看一下我的显卡。

`$ neofetch` 一下：

![neofetch](fetch.png)

看到了两个 GPU:

- NVIDIA GeForce MX250
- Intel UHD Graphics 620
- 等等为什么有两个？！:no_mouth:

上网搜了一下，是可以有双显卡的，一个集显一个独显。不做调整的话默认开机使用用的是核显，使用到一定量，会自动切换到独立显卡。不过也有些因为驱动问题，导致一直默认核显。（啊我好像没装驱动！:joy:）

### Intel UHD Graphics 620

`$ lspci | grep -i UHD` 看一下：

![Intel 显卡](UHD.png)

这就是那个集成显卡，是集成在 i5 的那个 CPU 里的。UHD不适用于硬核游戏。相反，它提供了用于管理笔记本电脑的屏幕和一些轻便游戏的基本性能。

它是 VGA compatible controller。

### NVIDIA GeForce MX250

这就是那个独立显卡，经典 Nvidia 的，`$ lspci | grep -i Nvidia` 看一下：

![Nvidia 显卡](Nvidia.png)

查了一下这个显卡也不咋地，只能说比集显好一点。功耗低性能差，游戏绘图设计就别考虑了。**彳亍**。

然后我意外地发现这个 Nvidia 的好像在这没装驱动，也就是说一直没用上。。。



# 装驱动

发现连显卡驱动都没有赶紧装一下:disappointed_relieved:

[找了一篇教程](https://phoenixnap.com/kb/install-nvidia-drivers-ubuntu)

[另一篇教程](https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/#verification)

直接GUI图形化安装吧。

1. 直接在软件搜索里搜 `driver` ，点那个绿的就跳出来了。

![找driver](find_friver.png)

2. 这里有好多种驱动，选一种安装。

![安装驱动](install_driver.png)

还好问了一下IT侠这几个的区别，前几个是nv做的闭源驱动，最后一个是开源的，建议上网搜一下看自己的卡用哪个驱动好。

这上面显示460是recommended，那就用它了。

![选这个](drivers.png)

安装成功之后，重启。

`$ nvidia-smi` 检查一下，出现这个界面就成功了。

![驱动安装成功](driver_succ.png)

`$ nvidia-setting` 打开驱动设置。

`$ sudo prime-select nvidia/intel` 可以切换显卡。

（有趣的是过了几天我的 `$ nvidia-smi` 突然打不开了，显示什么 `communicate failed`，解决方法是在 `intel` 和 `nvidia` 之间切换一下再重启就可以了。）



---

# 装 Cuda

终于一切准别就绪，开始安装cuda。

[官方文档](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

- GPU 一定要是支持 cuda 的。
- 还需要 `gcc`，应该都有吧。
- 可选安装一个 GDS。（我也不知道这是啥）
- 首先还要下一个 **CUDA Toolkit**，在[这](https://developer.nvidia.com/cuda-downloads)下，进去选择自己的设备型号按指示就可以了。
  - 遇到一个问题就是我们已经安装了 Nvidia Driver，所以在后面 sh xxx.run 时有一个安装 driver 的选项就不要选了，否则会失败。
  - 这也说明在 Cuda Toolkit 中会自带一个驱动从而不一定要先装显卡驱动？
  - 出现这个就是成功辣！![安装成功！](toolkit_succ.png)

- 最后设置一下环境变量，`$ sudo vim ~/.zshrc` 中添加：

  - `export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}` （注意这里的cuda-xx.x文件夹要看下载的是什么版本）

  - `export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`

到这里其实就差不多完成了，运行一下 `$ nvcc -V` 如下显示就可以了。

![nvcc](nvcc.png)

然后在 `/usr/local/cuda-11.4/samples` 中有一些测试用例，可以自己玩一玩。



## 一些问题

我在之后回看的时候发现了一些问题，在知乎上找到一篇[很好的文章](https://zhuanlan.zhihu.com/p/91334380)讲清楚了基本所有涉及到的概念。

- 一个是在上上那张图里的 warning 中说没有配上 driver，并且至少要 470 的驱动。
  - 但是我的 sample 中的样例都可以跑。。。所以暂时不严谨地认为 cuda 还是能用的（，驱动和cuda版本不兼容的问题暂时还没有暴露出来。
  - 不过我不知道跑 sample 是否需要 drive 的正确参与，如果不一定的话还是说明只有一个没有兼容上驱动的 cuda壳子。
  - 如果到时候出现了版本兼容的问题，考虑的解决方法是将 cuda 和驱动全卸了按照版本对应重装。这就提到了我在上面装 cuda 的那个步骤中，其实运行 `.run` 文件的时候就可以在那里装驱动了，它版本应该是帮我配好了的。
  - 其实电脑上是可以装上多个驱动和多个 cuda 的。不过多个驱动好像 ubuntu 上会崩，yls就这么翻车了（。多个 cuda 可以通过在 `$PATH` 中放那个 `...\cuda\bin` 路径而不是带版本号的 `...\cudax.x\bin`路径，然后通过 cuda 的符号链接来控制多个版本。不过一是为了避免复杂性二是我这电脑磁盘真不够了qwq，会考虑全卸了重装。

- 另一个问题其实是在装 `pytorch` 时遇到的，在 `conda install` 时会装一个也叫 `cudatoolkit` 的包，搞不懂这个和 Nvidia 官网下的有啥区别。并且 conda 装的 `pytorch` 和 `cudatoolkit` 版本都是11.1的，不知道和我那个大 cuda 又有什么区别。
  - 这个其实就在我上面提到的那个文章以及它引用的[另一篇](https://www.cnblogs.com/yhjoker/p/10972795.html)中有详细的阐释。总之就是 Nvidia 官方下的那个大 cuda 是个完整的工具包；而像 pytorch 这样的框架，只要用到 cuda 中的动态链接库。所以 conda 在下 `pytorch` 时 ，就会装一个 “mini cuda toolkit”（其实是和大 cuda 有重叠的），只要有和这个小 cuda 兼容的驱动，pytorch 就能跑。
  - 所以其实如果我们还没装大 cuda，`torch.cuda.is_avaliable()` 应该也是 `True`，`pytorch` 没有大开发需求的话基本可以正常用的。这也因为小 cuda 的版本是 11.1，应该是和我们 460 的驱动兼容的。

总之，大 cuda （用大小来外号这两个cuda好像也是个好方法:joy:）和 driver 好像就是没兼容上的，不过我们有兼容好的小 cuda，跑 `pytorch` 是基本没问题的。出了问题再说（挖坑）。



## 卸载

因为上述问题要重装所以再总结一下卸载流程。

cuda 有自带的卸载脚本

``` shell
$ sudo /usr/local/cuda/bin/cuda-uninstaller
```



---

# 装 cuDNN

[官方文档](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

首先需要一个nvidia的账号，自己注册一个。

首先要在[这](https://developer.nvidia.com/rdp/cudnn-download)下载cuDNN，完成一个调查表，点击同意协议就可以出现了，选择自己的版本的下载。

下载这几个：

![cudnn list](cudnn.png)



- 将 `cuDNN Library for Linux` 解压，复制到 cuda 环境中。

``` shell
$ tar -xzvf cudnn-x.x-linux-x64-v8.x.x.x.tgz
$ sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
$ sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
$ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

- 将 `.deb` 文件安装。

```shell
$ sudo dpkg -i libcudnn8_x.x.x-1+cudax.x_amd64.deb	# runtime library
$ sudo dpkg -i libcudnn8-dev_8.x.x.x-1+cudax.x_amd64.deb	# developer librarty
$ sudo dpkg -i libcudnn8-samples_8.x.x.x-1+cudax.x_amd64.deb	# samples and doc
```

- 检查一下。

```shell
$ cp -r /usr/src/cudnn_samples_v8/ $HOME
$ cd  $HOME/cudnn_samples_v8/mnistCUDNN
$ make clean && make
```

我遇到一个报错：

```shell
>>> WARNING - FreeImage is not set up correctly. Please ensure FreeImage is set up correctly. <<<
```

检查一下是一个包没有安装，装一下就行。

```shell
$ sudo apt-get install libfreeimage3 libfreeimage-dev
```

编译完之后运行一下可执行文件，出现 `Test passed!` 就说明 cuDNN 已经正确安装辣！:happy:



## 卸载

``` shell
$ rm /usr/local/cuda/include/cudnn.h
$ rm /usr/local/cuda/lib64/libcudnn*
```

然后还要把几个 `.deb` 包点开卸载，我看网上的教程都没有这一步，我觉得还是必要的因为版本不一样。

