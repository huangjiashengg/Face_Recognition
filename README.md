# 人脸识别

## 前言

本项目展现了从静态图像的人脸检测，到最终的人脸识别实现过程。其中涉及到的技术手段包括应用Haar级联特征数据抓取人的器官部位(眼睛、耳朵、鼻子、脸等)，加载训练数据进行数据训练，以便进行人脸识别等。应用到的模块：opencv-contrib-python, opencv-python

## 静态图像人脸检测

1，相关实现过程放在‘人脸检测.py’文件中；

2，主要思路：利用级联特征数据，对图像进行人脸检测之后，画出检测出的矩形区域可视化，从而得到人脸检测结果

## 视频人脸检测

1，相关实现过程放在‘人脸检测.py’文件中；

2，主要思路：利用级联特征数据，对视频进行人脸检测之后，画出检测出的矩形区域可视化，从而得到人脸检测结果

3，视频其实是由一帧一帧的图像组成的，因此加载视频进行人脸检测主要在静态图像人脸检测的基础之上进行实现

## 人脸识别数据训练

1，相关实现过程放在‘人脸识别-数据训练.py’文件中；

2，先检测，再训练；

3，基于LBPH算法进行图像训练；

4，训练的目的是为识别做铺垫。

## 人脸识别应用

1，加载训练后得到的训练文件；

2，输入一个新的图像，人脸识别输出对应的之前用于训练图像中的id，以及相应的置信评分；

3，置信评分越低，表明识别吻合程度较高；0代表完全匹配，任何高于80分的置信评分都会被认为吻合程度较低。

## 交流

有问题随时联系2393946196@qq.com，欢迎互相学习交流！