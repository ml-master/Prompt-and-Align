# Prompt-and-Align
Implementation of Prompt-and-Align: Prompt-Based Social Alignment for Few-Shot Fake News Detection, ACM Conference on Information and Knowledge Management (CIKM) 2023. Jiaying Wu, Shen Li, Ailin Deng, Miao Xiong, Bryan Hooi. (https://arxiv.org/abs/2309.16424)

## Data
数据的获取方式请参照[P&A](https://github.com/jiayingwu19/Prompt-and-Align)

**从原始社交背景构建新闻邻接图** 

作为使用“data/adjs/”下预处理邻接矩阵的替代方案，在“Process/adj_matrix_fewshot.py”中提供了一个预处理脚本来从头开始构建矩阵

使用以下命令构造邻接矩阵：

```bash
mkdir data/adjs_from_scratch
python Process/adj_matrix_fewshot.py
```

## 运行 Prompt-and-Align

P&A的源码在 `prompt_and_align.py`. 

通过以下命令执行:

```bash
sh run.sh
```

## Experiment
| 数据集 | PolitiFact | GossipCop | FANG |
|---------|------------|-----------|------|
| 16      | 0.8074     | 0.6914    | 0.6038|
| 32      | 0.8457     | 0.6781    | 0.6179|
| 64      | 0.8727     | 0.6529    | 0.6401|
| 128     | 0.8885     | 0.8224    | 0.6698|

| 数据集 | PolitiFact | GossipCop | FANG  |
|---------|------------|-----------|-------|
| 16      | 0.6847     | 0.7194    | 0.3058|
| 32      | 0.8531     | 0.7549    | 0.4556|
| 64      | 0.8439     | 0.8531    | 0.4959|
| 128     | 0.8781     | 0.9854    | 0.5187|

| 数据集 | PolitiFact | GossipCop | FANG  |
|---------|------------|-----------|-------|
| 16      | 0.9301     | 0.6634    | 0.9017|
| 32      | 0.8383     | 0.6012    | 0.7802|
| 64      | 0.9015     | 0.4528    | 0.7843|
| 128     | 0.8988     | 0.6595    | 0.7818|
