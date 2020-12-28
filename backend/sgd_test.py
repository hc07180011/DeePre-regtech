import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier # 下次做投影片講這個
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # 下次做投影片講這個

from matplotlib.font_manager import FontProperties
myfont = FontProperties(fname=r'./taipei_sans_tc_beta.ttf')


train_title = ['圓方2013', '圓方2014', '圓方2016', '長鴻2012', '長鴻2013', '長鴻2014']
train_X = np.array([np.nan_to_num(pd.read_excel('train.xlsx', sheet_name=i, header=None).to_numpy()).flatten() for i in range(2, 8, 1)])
train_Y = np.array([0, 0, 1, 0, 0, 1])


# clf = make_pipeline(StandardScaler(),
#                     SGDRegressor(max_iter=100, verbose=1))
# a = clf.fit(train_X, train_Y)

# labels = np.nan_to_num(pd.read_excel('train.xlsx', sheet_name=0, header=None).to_numpy())[:,:14].flatten()

# weights = [[x for x, y in zip(labels, a.named_steps['sgdregressor'].coef_) if not (x!=x) and y], [round(y, 2) for x, y in zip(labels, a.named_steps['sgdregressor'].coef_) if not (x!=x) and y]]

# plt.figure(figsize=(16, 4))
# plt.plot(np.arange(len(weights[0])), weights[1], marker='o', markersize=3)
# plt.plot(np.arange(len(weights[0])), np.zeros(len(weights[0])), 'r--')
# plt.xticks(np.arange(len(weights[0])), [x if i%2 else '\n'+str(x) for (i, x) in enumerate(weights[0])], fontproperties=myfont, fontsize=8)
# plt.savefig('weight.png')


pca = PCA(n_components=2)
pca.fit(train_X)

# df = pd.DataFrame(pca.components_, columns=labels, index=['PC-1','PC-2'])
# df.to_excel('tmp.xlsx')

clf = make_pipeline(StandardScaler(),
                    SGDRegressor(max_iter=100, verbose=1))
a = clf.fit(pca.transform(train_X), train_Y)

points = np.array(list(itertools.product(np.arange(-15, 16, 0.1), np.arange(-5, 16, 0.1))))
preds = a.predict(points)

cs = []
for p, pred in zip(points, preds):
    weight = max(0, min(255, int(pred * 255)))
    print('#{:02x} {:02x} {:02x}'.format(weight, weight, weight))
    cs.append('#{:02x}{:02x}{:02x}'.format(weight, weight, weight))

for idx, point in enumerate(pca.transform(train_X)):
    plt.scatter(point[0], point[1], c=('b' if train_Y[idx] == 0 else 'r'))
    plt.annotate(train_title[idx], (point[0], point[1]), c='#ffffff' if train_Y[idx] == 0 else 'b', fontproperties=myfont)

plt.scatter(points[:, 0], points[:, 1], s=1, c=cs, alpha=0.5)


plt.savefig('test.png')
