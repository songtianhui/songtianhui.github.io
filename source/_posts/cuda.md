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

















