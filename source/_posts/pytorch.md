---
title: pytorch 安装教程
date: 2021-07-20 15:35:56
tags:
typora-root-url: pytorch
---

之前已经安装好了 cuda，现在来安装 pytorch，顺便再装一个 anaconda。同样的，本篇是针对 Ubuntu 

20.04 LTS 的 pytorch 和 anaconda 安装流程。



# 安装 anaconda

以前一直没搞清已经安装的 python pip 和 anaconda 有什么区别，会不会有冲突，所以一直没敢装 anaconda，现在好像弄清楚了，它们基本上是两个独立的程序，conda会使用它自己的虚拟环境，所以影响应该不大，那我们就开始吧。

[官方教程](https://docs.anaconda.com/anaconda/install/linux/)

首先到[官网下载](https://www.anaconda.com/products/individual#Downloads)安装包，可以用 `sha256sum` 检查一下。

然后到安装路径运行它：

![installer](installer.png)

然后有一些协议啊步骤啊，阅读同意就行。有一个初始化选项推荐 `yes`。

出现这个就是安装完成辣：

![install successfully!](install_succ.png)

看看 `python`，发现已经是 anaconda 的了。

![python after installation](python-conda.png)

改一下清华源：

```shell
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
$ conda config --set show_channel_urls yes
```

`$ anaconda-navigator` 可以打开 anaconda 的 gui 界面，不过好像窗口大小一直有问题。

`$ conda update conda` 可以进行更新。



# 安装 pytorch

到 [pytorch 官网](https://pytorch.org/) 直接选择自己的电脑版本，然后输入他给的命令行就行了。

![](pytorch_org.png)

出现这个就是开始下载了，有一点慢：

![](pytorch_succ.png)

