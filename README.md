# A Semantic and Structural Transformer for Code Summarization Generation

This is the implementation for A Semantic and Structural Transformer for Code Summarization Generation

## 1. Install environments

Use the following command

```
pip install -r requirements.
```

## 2.Pre process data & Meteor

Our data processing is the based on [1],

We also share the 'paraphrase-en.gz' in valid_metrices/meteor/

The Baidu Netdisk and Google Drive link are as follows

baidu link: https://pan.baidu.com/s/1EqDA3zh2fmfRAbQPtmrbfQ 

pwd: 1111

google link: https://drive.google.com/drive/folders/1ZnmxMRVrlXefMkEuSl93H8erPs6QGWH3?usp=sharing

## 3. Train

The config of model is saved at ./config, you can change 'data_dir' to your own data set path, the "ast_trans_for_java.py" and "ast_trans_for_py.py" is for training.

For train the model:

single-gpu:

```
export CUDA_VISIBLE_DEVICES=0
python main.py --config ./config/ast_trans_for_py.py --g 0
```

multi-gpu:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --config=./config/ast_trans_for_py.py --g 0,1,2,3
```


## 4. Test

For test, you can set the parameter 'is_test' to True and set the checkpoint file path.
The program will default find the checkpoint based on the hype-parameters.


```
export CUDA_VISIBLE_DEVICES=0
python main.py --config ./config/ast_trans_for_py_test.py --g 0
```


# Acknowledgement

Our module and dataset processing work is based on these two papers, and we salute the dedicated researchers here

[1]Tang Z, Shen X, Li C, et al. AST-trans: Code summarization with efficient tree-structured attention[C]//Proceedings of the 44th International Conference on Software Engineering. 2022: 150-162.

[2]Wang Z, Ma Y, Liu Z, et al. R-transformer: Recurrent neural network enhanced transformer[J]. arXiv preprint arXiv:1907.05572, 2019.
