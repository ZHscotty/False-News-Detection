# False-News-Detection
假新闻识别的一个比赛
比赛地址：
https://biendata.com/competition/falsenews/


1.position embedding
2.cnn + rnn + (attention)
3.bert
4.attention(新的尝试，间距！！！！)
5.seqlen使用
6.分句来做，层次的attention
7.embedding那里结合双向的LSTM形成新的embedding（https://zhuanlan.zhihu.com/p/25928551）

效果记录：
1.bilstm 0.76
1.bisltm+att(填充方式为pre) 测试集效果0.79
2.bilstm+att(填充方式为post) 测试集效果0.77
3.bilstm+att(发现之前bilstm后没加激活和dropout，加上后) 测试集效果0.77
4.cnn+bilstm+att 测试集效果0.70
5.bilstm+att(使用seqlen过滤padding的值) 测试集效果0.72
6.使用训练集训练词向量(bilstm+att(使用seqlen过滤padding的值)) 测试集效果0.73
7.字级别
8.DGCNN
9.posotion embedding + attention + lstm
10.lstm + att + lstm
11.cnn 0.78
12.bert 0.89