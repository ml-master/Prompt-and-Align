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

