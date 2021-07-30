This directory contains code and data for the ACL 2021 paper [**How Knowledge Graph and Attention Help?
A Quantitative Analysis into Bag-level Relation Extraction**](https://arxiv.org/abs/2107.12064). Our implementation is based on THUNLP's [RE-Context-or-Names](https://github.com/thunlp/RE-Context-or-Names).

### 1 Dataset

We provide preprocessed NYT-FB60K in `data/`, it has `train.txt` and `test.txt`. There is no development set for NYT-FB60K.

Please download the `nyt.zip` from [google drive](https://drive.google.com/file/d/1kuqaiebhNnatccUB4aLVSrX1UFuNMabZ/view?usp=sharing) and put it under `data/`, then unzip it.


### 2 Train

Run the following scirpt:

```shell
cd code/nyt
bash train.sh
```

If you want to skip the training time, you can download our finetuned model `nyt_bert-base-uncased_TransE_re_direct__kg.mdl` from [google drive](https://drive.google.com/file/d/1gYDboKHbR108Iulk_1_-HqsMDjNvt5r3/view?usp=sharing). and put it under `save/nyt/`.

### 3 Test

After you get finetuned model, please run the following scirpt:

```shell
cd code/nyt
bash test.sh
```
