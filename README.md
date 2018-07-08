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

## config
As you can see, the script refers config file.
Thus, you change your settings and try to check the situation.


The explanation of each parameter is follows:


# Authors
Yoshiaki Kitagawa
