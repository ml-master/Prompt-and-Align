# Prompt-and-Align
Implementation of Prompt-and-Align: Prompt-Based Social Alignment for Few-Shot Fake News Detection, ACM Conference on Information and Knowledge Management (CIKM) 2023. Jiaying Wu, Shen Li, Ailin Deng, Miao Xiong, Bryan Hooi. (https://arxiv.org/abs/2309.16424)

## Data
数据的获取方式请参照[P&A](https://github.com/jiayingwu19/Prompt-and-Align)

**从原始社交背景构建新闻邻接图** 

As an alternative to using our pre-processed adjacency matrices under `data/adjs/`, we provide a pre-processing script at `Process/adj_matrix_fewshot.py` to construct the matrices from scratch.  

Construct the adjacency matrices with the following command:

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

