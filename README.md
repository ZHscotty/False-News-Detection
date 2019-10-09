# False-News-Detection
假新闻识别的一个比赛
比赛地址：
https://biendata.com/competition/falsenews/

数据分析：


效果记录：
1.单用bilstm或者bilstm+att 效果在0.77左右
2.cnn+bilstm+att 测试集效果0.70(可能是cnn参数设置的问题)
3.单用cnn 效果0.78
4.单用bert 效果0.89
5.cbow+cnn 效果0.82
6.bert+bisltm+att 0.88
7.bert+cnn+att 0.88
8.bert+biattention 0.87
9.对三个bert后的结果投票融合结果 0.88
