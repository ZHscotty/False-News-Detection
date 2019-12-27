# False-News-Detection
> 假新闻识别比赛-[比赛地址](https://biendata.com/competition/falsenews/)

## 数据分析：


## 效果记录：
| 实验方法 | 线上分数 | Model |
| --- | --- | --- |
| BILSTM(BILSTM+ATT) | 0.77 | [MODEL]()
| CNN | 0.78 | [MODEL]()
| BERT | 0.89 | [MODEL]()
| CBOW+CNN | 0.82 | [MODEL]()
| BERT+BILSTM+ATT | 0.88 | [MODEL]()
| BERT+CNN+ATT | 0.88 | [MODEL]()
| BERT+BIATT | 0.87 | [MODEL]()
| 三种BERT模型融合 | 0.88 | [MODEL]()

## 心得感悟：
1. 没有充分利用Bert得到的字向量，过度依赖于Bert-Finetune
2. 没有思考过Bert训练出的字向量和词向量的结合，作为输入
3. text-CNN没有利用好
4. 交叉验证的重要性
