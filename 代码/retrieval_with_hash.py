import time
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
from DHash import DHash

plt.rcParams['font.sans-serif'] = 'SimHei'

(x_train, _), (x_test, _) = fashion_mnist.load_data()

# 图像归一化
x_test = x_test.astype('float32') / 255.
x_test = np.reshape(x_test, (-1, 28, 28, 1))

# 为了计算查全率和召回率，人为创造相同图片作为相似图片
# 例如0和10000，20000，30000就是相同图片
x_test = np.concatenate((x_test, x_test), axis=0)
x_test = np.concatenate((x_test, x_test), axis=0)
print(x_test.shape)

# 载入训练好的编码模型
encode = load_model('encoder.h5')

# 设置检索图像下标
NUM = 18

# 开始不带哈希的检索测试
query = x_test[NUM]
# 展示待检图片
plt.imshow(query.reshape(28, 28), cmap='gray')
plt.show()

# 对剩余测试集图片提取特征
test_features = encode.predict(x_test)

# 对待检图片提取特征
query_features = encode.predict(query.reshape(1, 28, 28, 1))
test_features = test_features.reshape(-1, 4 * 4 * 8)
query_features = query_features.reshape(1, 4 * 4 * 8)
# print(test_features[-1].shape,query_features.shape)
# 对提取后的特征进行hash编码
test_hash = []
for i in range(len(test_features)):
    test_hash.append(DHash.calculate_hash(test_features[i]))

# 得到最相似的x个图片
n_neigh = [1, 3, 5, 9, 10]

base_list = [0, 10000, 20000, 30000]
base_list = [i + NUM for i in base_list]

match_num = 0

precisions = []
recalls = []


# 检索时间
start = time.time()
query_hash = DHash.calculate_hash(query_features)

for neigh in n_neigh:
    distances = []
    for i in range(len(test_hash)):
        distances.append(DHash.hamming_distance(query_hash, test_hash[i]))
    res = sorted(enumerate(distances), key=lambda x: x[1])
    indices = res[:neigh]
    indices = [i[0] for i in indices]
    closest_images = x_test[indices]
    print(f"检索耗时:{time.time() - start}s")

    closest_images = closest_images.reshape(-1, 28, 28, 1)

    # 展示最相似的x张图
    plt.figure(figsize=(20, 6))
    for i in range(neigh):
        ax = plt.subplot(1, neigh, i + 1)
        plt.imshow(closest_images[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    print(f"待检图片下标:{NUM}")
    print(f'检索图片下标：{indices}')

    match_num = len([i for i in list(indices) if i in base_list])

    # 查准率
    precisions.append(match_num / neigh)
    # 召回率 有四组重复数据
    recalls.append(match_num / 4)


    # 更新第二次检索时间
    start = time.time()

# AP 查询检索精度
AP = sum(precisions) / len(n_neigh)

print(f"测试检索{len(n_neigh)}次，查准率分别为:{precisions}\n"
      f"召回率分别为:{recalls}\n"
      f"查询检索精度:{AP}")

# 绘制PR曲线
plt.figure()
plt.plot(recalls, precisions)
plt.title("PR曲线")
plt.xlabel("召回率")
plt.ylabel("查准率")
plt.show()

# # 绘制MAPs
# plt.figure()
# plt.plot(n_neigh, MAPs, c='r')
# plt.xlabel("topk")
# plt.ylabel("MAP")
# plt.xticks(n_neigh,n_neigh)
# plt.title("MAPs")
# plt.show()
