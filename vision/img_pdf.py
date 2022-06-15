import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')
plt.rcParams.update({"font.size":20})
# plt.switch_backend('agg')

import sys
sys.path.append('.')


# df = pd.read_csv('glcm_features.csv')
# df.head()

# array = df.values
# x = array[:,2:]
# y = array[:,1].astype('int')
# print(x.shape)
# print(y.shape)

import seaborn as sns
def plotConfusionMatrix(values, title, labels, figsize=(6, 6)):
    con_mat_df = pd.DataFrame(values, index=labels, columns=labels)
    figure = plt.figure(figsize=figsize)
    sns.set(font_scale=1.2)
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues, fmt='g', annot_kws={"size": 14})
    plt.tight_layout()
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    figure.savefig('confusion_matrix.png')
    # plt.savefig('results.png')


def plot(matrix, num_classes, labels):
    matrix = matrix
    # print(matrix)
    fig = plt.figure('ret')
    plt.imshow(matrix, cmap=plt.cm.Blues)

    # 设置x轴坐标label
    plt.xticks(range(num_classes), labels, rotation=45)
    # 设置y轴坐标label
    plt.yticks(range(num_classes), labels)
    # 显示colorbar
    plt.colorbar()
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('SVM')

    # 在图中标注数量/概率信息
    thresh = matrix.max() / 2
    for x in range(num_classes):
        for y in range(num_classes):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(matrix[y, x])
            plt.text(x, y, info,
                        verticalalignment='center',
                        horizontalalignment='center',
                        color="white" if info > thresh else "black")
    plt.tight_layout()
    plt.show()
    # fig.savefig('svm.pdf', dpi=1200, format='pdf')

matrix = np.array([[1,2,3],[1,2,4],[1,2,3]])
surface_type_labels = ["wet", "dry", "icy"]
# plotConfusionMatrix(matrix, "SVM", surface_type_labels)

plot(matrix, 3, surface_type_labels)

