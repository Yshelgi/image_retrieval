from model_AE import autoencoder
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
# 开始训练自编码器

# 载入数据集
(x_train, _), (x_test, _) = fashion_mnist.load_data()

# 图像归一化
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


x_train = np.reshape(x_train, (-1, 28, 28, 1))
x_test = np.reshape(x_test, (-1, 28, 28, 1))


Epoches=20

# 编译训练模型
autoencoder.compile(optimizer='adam', loss='mse')
history = autoencoder.fit(x_train, x_train, epochs=Epoches, batch_size=32)

# 查看训练的损失曲线
plt.figure()
plt.plot(range(Epoches), history.history['loss'])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 保存训练好的模型,主要是编码器用于未来对图像特征提取
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
encoder.save('encoder.h5')
