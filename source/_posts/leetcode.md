---
title: leetcode 刷题笔记
date: 2021-07-05 23:38:08
tags:
---

问求上完了，依然不能放松对数据结构和算法的训练，要坚持不懈的刷题。着重对一些经典题型和方法进行总结。

<!--more -->

# 基本常见算法

## 两数之和

> 给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** `target` 的那 **两个** 整数，并返回它们的数组下标。

首先肯定有一个 $O(n^2)$ 的朴素解法，然后我想到先排序然后用首尾指针向中间移动，$O(n\log{n})$。但是需要一个哈希来存排序前的下标，既然这样其实就可以直接用哈希 $O(n)$ 解了。

``` c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int n = nums.size();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (nums[i] + nums[j] == target) {
                    return {i, j};
                }
            }
        }
        return {};
    }
};
```





# 剑指offer系列

## 04. 二维数组中查找

> 在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

原来是想的二分查找但是思路很明显是错的。只要从一个角落开始，每个元素比较，如果不一样就超朝另外两个方向移动。

其实也是一个压缩解空间的思想，比如我从右上角开始，当前元素比target大，说明在该元素下面所有的元素都不符合，只能向左移动一位。

注意边界条件`matrix = []`。复杂度 $O(m + n)$。

``` c++
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        int len1 = matrix.size();
        if (len1 == 0) return false;
        int len2 = matrix[0].size();
        int h = 0, v = len2 - 1;
        while (h < len1 && v >= 0) {
            int p = matrix[h][v];
            if (p == target) return true;
            if (p > target) {
                v--;
                continue;
            }
            if (p < target) {
                h++;
                continue;
            }
        }
        return false;
    }
};
```
