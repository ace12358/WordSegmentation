# WordSegmetation
This is Japanese Word Segmentation tool using LSTM.

[Screen Shot]

# Dependency
```
python==3.6.2
chainer==1.4.0
filelock==2.0.4
nose==1.3.7
numpy==1.10.1
protobuf==2.6.1
six==1.10.0
```
# Setup
```
pip install -r requirements.txt
```
# Usage
## Train
moving to `src` directory, you can execute training test.

```
python3 train_word_segmentater.py ../configs/test_config.ini
```
This script execute training and evaluating.
The training results is written a `results/` directory (that is based on config)


## config
As you can see, the script refers config file.
Thus, you change your settings and try to check your situation.

#### config
The explanation of each parameter is follows:
| Section    | Item                  | Explanation                                                                           |
|------------|-----------------------|---------------------------------------------------------------------------------------|
| Data       | train                 | train file. Each word must be separated by ' ' (space)                                |
| Data       | test                  | test file. this is raw corpus (is not separated).                                     |
| Data       | dict                  | dict file to create dict feature. One word per line.                                  |
| Settings   | label_num             | 2, 3, 4 stands for 'BI', 'BIE', and 'BIES' respectively.                              |
| Settings   | batch_size            | this program update weight per batch size. (ex. '30' means updating per 30 sentences) |
| Settings   | n_epoch               | epoch number. the program read traing file `n_epoch` times                            |
| Parameters | window                | window size. the number of character incorporate to input layer at label prediction.  |
| Parameters | embed_units           | character embeddings dimension.                                                       |
| Parameters | char_type_embed_units | character type embeddings dimension.                                                  |
| Parameters | hidden_units          | hidden layer dimension                                                                |
| Parameters | learning_rate         | learning rate for optimizer.                                                          |
| Parameters | lam                   | weight decay parameter for L2 regularization.                                         |
| Result     | raw                   | the result of test file segmentated by training model                                 |
| Result     | config                | this config file setting output the directory that you specify.                       |
| Result     | model                 | trained model output the directory that you specify.                                  |
| Result     | evaluation            | Precision, Recall, and F1-score output the directory that you specify.                |

# Authors
Yoshiaki Kitagawa
