---
title: 计算机网络课程笔记-第七章无线网络和移动网络
date: 2021-09-18 17:38:57
tags:
categories:
- 课程
- Network
mathjax: true
---

# 概述

- 无线主机（wireless host）
  - 手机等，可能移动也可能不移动。
- 无线通信链路（wireless communication link）

<!--more-->

- **基站（base station）**
  - 通常负责协调与之相关联的多个无线主机的传输。
  - 蜂窝塔（cell tower），无线LAN的接入点（access point）。
- 移动主机可能需要改变与之相关联的基站，叫**切换（handoff）**。
- 网络基础设施。无线主机进行通信的更大的网络。



# 无线链路和网络特征

- 递减的信号强度。（路径损耗 path loss）
- 来自其他源的干扰。
- 多径传播（multipath propagation）。

所以无线链路中比特差错比有线链路中更为常见，不仅CRC，还要ARQ协议。

- **信噪比（Signal-to-Noise Rate, SNR）**是所收到的信号和噪声强度的相对测量。单位通常是分贝dB。
- **比特差错率（Bit Error Rate, BER）**。

一些特征：

- 对于给定的调制方案，SNR越高，BER越低。
- 对于给定的SNR，具有较高比特传输率的调制技术将具有较高的BER。
- 物理层调制技术的动态选择能用于适配对信道条件的调制技术。



# WiFi: 802.11 无线 LAN

我们常用的无线网就是 **IEEE 802.11 无线 LAN**，也就是所说的 **WiFi**。

## 802.11 体系结构

- **基本服务集（Basic Service Set, BSS）**。
  - 一个 BSS 包含一个或多个无线站点和一个称为接入点的中央基站。
- 配置 AP 的无线 LAN 经常被称作**基础设施无线LAN（Infrastructure wireless LAN）**。

### 信道与关联

- 当网络管理员安装一个 AP 时，管理员为该接入点分配一个单字或双字的**服务集标识符（Service Set Identifier, SSID）**。管理员还必须为该 AP 分配一个信道。
- **WiFi 丛林（jungle）**是一个任意物理位置，在这里无线站点能从两个或多个 AP 中收到很强的信号。
- 为了获得因特网接入，无线站点需加入其中一个子网并因此需要与其中一个 AP 相**关联（associate）**。
- 每个 AP 需周期性地发送**信标帧（beacon frame）**，包括该 AP 的 SSID 和 MAC地址。
- 扫描信道和监听信标帧的过程被称为**被动扫描（passive scanning）**。无线主机也能执行**主动扫描（active scanning）**。



### 802.11 MAC 协议

使用随机访问协议——**带碰撞避免的 CSMA（CSMA with Collision avoidance, CSMA/CA）**。和以太网的 CSMA/CD 相似。

- 链路层确认（link-layer acknowledgement）方案
  - 目的站点收到一个通过 CRC 校验的帧后，它等待一个被称作**短帧间间隔（Short Inter-Frame Spacing, SIFS）**的一小段时间，然后发回一个确认帧。
- CSMA/CA协议
  - 如果某站点最初监听到信道空闲，它将在一个被称作**分布式帧间间隔（Distributed Inter-Frame Space, DIFS）**的短时间段后发送该帧。
  - 否则，该站点选取一个随机回退值并且在侦听信道空闲时递减该值。当侦听到信道忙时，计数值保持不变。
  - 当计数值减为0时，该站点发送整个数据帧。
  - 如果收到确认，发送站点知道它的帧已被目的站点正确接收了。如果该站点要发送另一帧，它将从第二步开始 CSMA/CA 协议。如果未收到确认，发送站点将重新进入第二步中的回退阶段，并从一个更大的范围内选取随机值。
- **请求发送（Request to Send, RTS）**
- **允许发送（Clear to Send, CTS）**
- **点协调功能（Point Coordination Functino, PCF）**
  - **PIFS**
  - 超级帧


# MAC 帧
- 管理帧（Management Frames）
- 控制帧（Control Frames）
  - Power Save-Poll(PS-Poll)
  - Contention-Free-End(CF-End)
- 数据帧（Data Frames）
  - Data/+CF-Ack/+CF-Poll/+CF-Ack+CF-Poll
  - Null Function（空帧）
  - CF-Ack/CF-Poll/CF-Ack+CF-Poll

