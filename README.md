# CCFDF-Personalized-Matching-Model-of-Packages-for-Telecom-Users
- Aimed at different users, how to recommend personalized telecommunication packages is a multi-class problem essentially.
- We extract users' demographic features, behavior features and preference through feature project. In addition, we training random forest (RF), XGBoost and deep network to classify.
- Expirements shows XGBoost has higher F1 values while RF has shorter training time.

**You can download full report from [here](https://github.com/PrideLee/CCFDF-Personalized-Matching-Model-of-Packages-for-Telecom-Users/blob/master/CCFBDCI2018-%E9%9D%A2%E5%90%91%E7%94%B5%E4%BF%A1%E8%A1%8C%E4%B8%9A%E5%AD%98%E9%87%8F%E7%94%A8%E6%88%B7%E7%9A%84%E6%99%BA%E8%83%BD%E5%A5%97%E9%A4%90%E4%B8%AA%E6%80%A7%E5%8C%96%E5%8C%B9%E9%85%8D%E6%A8%A1%E5%9E%8B.pdf).**

**You can download the raw data from [here](https://www.datafountain.cn/competitions/311/datasets).**

**You can download all the materials from [here](https://pan.baidu.com/s/14moLkACXh3iYjHsMhhxSTw).**

## CCFBDCI2018-面向电信行业存量用户的智能套餐个性化匹配模型
&emsp;&emsp;电信产业作为国家基础产业之一，覆盖广、用户多，在支撑国家建设和发展方面尤为重要。随着互联网技术的快速发展和普及，用户消耗的流量也成井喷态势，近年来，电信运营商推出大量的电信套餐用以满足用户的差异化需求，面对种类繁多的套餐，如何选择最合适的一款对于运营商和用户来说都至关重要，尤其是在电信市场增速放缓，存量用户争夺愈发激烈的大背景下。针对电信套餐的个性化推荐问题，通过数据挖掘技术构建了基于用户消费行为的电信套餐个性化推荐模型，根据用户业务行为画像结果，分析出用户消费习惯及偏好，匹配用户最合适的套餐，提升用户感知，带动用户需求，从而达到用户价值提升的目标。

&emsp;&emsp;套餐的个性化推荐，能够在信息过载的环境中帮助用户发现合适套餐，也能将合适套餐信息推送给用户。解决的问题有两个：（1）信息过载问题和用户无目的搜索问题。（2）各种套餐满足了用户有明确目的时的主动查找需求，而个性化推荐能够在用户没有明确目的的时候帮助他们发现感兴趣的新内容。通过利用已有的用户属性(如个人基本信息、用户画像信息等)、终端属性(如终端品牌等)、业务属性、消费习惯及偏好匹配用户最合适的套餐，对用户进行推送，完成后续个性化服务。该问题本质上是一个多分类问题，因此任何适用于多分类问题的模型算法均可用于该问题中。

<div align=center><img width="800" height="700" src="https://github.com/PrideLee/CCFDF-Personalized-Matching-Model-of-Packages-for-Telecom-Users/blob/master/age.png"/></div>

<div align=center><img width="800" height="700" src="https://github.com/PrideLee/CCFDF-Personalized-Matching-Model-of-Packages-for-Telecom-Users/blob/master/hot_map.png"/></div>

<div align=center><img width="800" height="700" src="https://github.com/PrideLee/CCFDF-Personalized-Matching-Model-of-Packages-for-Telecom-Users/blob/master/score.png"/></div>
