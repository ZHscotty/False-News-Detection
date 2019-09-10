# False-News-Detection
假新闻识别的一个比赛
比赛地址：
https://biendata.com/competition/falsenews/


1.position embedding
2.cnn + rnn + (attention)
3.bert
4.attention(新的尝试，间距！！！！)
5.seqlen使用


效果记录：
1.bisltm+att(填充方式为pre) 测试集效果0.79
2.bilstm+att(填充方式为post) 测试集效果0.77
3.bilstm+att(发现之前bilstm后没加激活和dropout，加上后) 测试集效果0.77
4.cnn+bilstm+att 测试集效果0.70
5.bilstm+att(使用seqlen过滤padding的值)