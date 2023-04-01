---
layout: publication
title: "[UIST'21] TypeBoard: Identifying Unintentional Touch on Pressure-Sensitive Touchscreen Keyboards"
subtitle: 'Yizheng Gu, Chun Yu*, Xuanzhong Chen, <b>Zhuojun Li</b>, Yuanchun Shi'
author: "CaveSpider"
header-style: text
---

![Teaser](/img/2021-10-10/typeboard_teaser.jpg)

In this work, we aim to address two common problems associated with tablet text entry.
Firstly, users often experience fatigue due to the inability to rest their hands on the tablet while typing.
Second, users may trigger unwanted inputs by accidental touches.
To tackle these issues, we collected typing data from 16 users, labeled each touch as intentional or not,
and trained an SVM model to accurately recognize unintentional touches (accuracy = 98.88%).
Our evaluation showed that TypeBoard effectively reduces fatigue and improves input speed.
It it also worth noting that this work is my first project at [Tsinghua HCI Lab](https://pi.cs.tsinghua.edu.cn).

## Paper Abstract

Text input is essential in tablet computer interaction.
However, tablet software keyboards face the problem of misrecognizing unintentional touch,
which afects efciency and usability.
In this paper, we proposed TypeBoard, a pressure-sensitive touchscreen keyboard that prevents unintentional touches.
The TypeBoard allows users to rest their fngers on the touchscreen, which changes the user behavior:
on average, users generate 40.83 unintentional touches every 100 keystrokes.
The TypeBoard prevents unintentional touch with an accuracy of 98.88%.
A typing study showed that the TypeBoard reduced fatigue (p < 0.005) and typing errors (p < 0.01),
and improved the touchscreen keyboard’ typing speed by 11.78% (p < 0.005).
As users could touch the screen without triggering responses,
we added tactile landmarks on the TypeBoard, allowing users to locate the keys by the sense of touch.
This feature further improves the typing speed, outperforming the ordinary tablet keyboard by 21.19% (p < 0.001).
Results show that pressure-sensitive touchscreen keyboards can prevent unintentional touch,
improving usability from many aspects, such as avoiding fatigue, reducing errors, and mediating touch typing on tablets.

## External Links

[[Paper]](/paper/TypeBoard.pdf)
[[DOI]](https://doi.org/10.1145/3472749.3474770)