# -*- encoding: utf-8 -*-
'''
@version : ??
@author  : jxw
@software: PyCharm
@file    : TASK01.py
@time    : 2019/6/21 下午8:36
'''

#%%
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)
#%%
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(len(train_data[0]), len(train_data[1]), len(train_data[2]))
print(len(train_data))
print(train_data[0])
print(train_labels[0], train_labels[1], train_labels[2])
#%%
# 字典：将数字转为单词
word_index = imdb.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0  # 用来将每一个sentence扩充到同等长度
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["UNUSED"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#%%
# 转译为原句
def decode_review(text):
    return (' '.join([reverse_word_index.get(i, '?') for i in text]))


decode_review(train_data[0])


'''
影评（整数数组）必须转换为张量，然后才能馈送到神经网络中。 我们可以通过以下两种方法实现这种转换：

1.对数组进行独热编码，将它们转换为由 0 和 1 构成的向量。
例如，序列 [3, 5] 将变成一个 10000 维的向量，除索引 3 和 5 转换为 1 之外，其余全转换为 0。
然后，将它作为网络的第一层，一个可以处理浮点向量数据的密集层。
不过，这种方法会占用大量内存，需要一个大小为 num_words * num_reviews 的矩阵。

2.可以填充数组，使它们都具有相同的长度，然后创建一个形状为 max_length * num_reviews 的整数张量。
我们可以使用一个能够处理这种形状的嵌入层作为网络中的第一层。
'''

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)


print(len(train_data[0]), len(train_data[1]), len(train_data[2]))
#%%
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000
'''
构建模型

输入数据是单词组合，标签是0或者1 先进行数据稀疏稠密化，
因为sequence里面的word_index值是[0~10000]内稀疏的，所以将每一个单词用一个16维的向量代替；
input(1024,256)output(1024,256,16) 
再通过均值的池化层，将每一个sequence做均值，类似于将单词合并 ;input(1024,256,16),output(1024,16) 
全连接层采用relu激活函数;input(1024,16),output(1024,16) 
全连接层采用sigmoid激活函数；input(1024,16),output(1024,1)
'''

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()
#%%
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)



results = model.evaluate(test_data, test_labels)

print('结果是：{}'.format(results))
#%%
history_dict = history.history
history_dict.keys()
#%%
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
#%%
plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
#%%
