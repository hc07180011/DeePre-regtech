import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from sklearn.linear_model import SGDClassifier # 下次做投影片講這個
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # 下次做投影片講這個

from matplotlib.font_manager import FontProperties
myfont = FontProperties(fname=r'./taipei_sans_tc_beta.ttf')

df = pd.read_excel('train_timeseries.xlsx', index_col=None)
feature_names = df.columns
company_names = df.to_numpy()[:,0]
train_X = df.to_numpy()[:,1:-1]
train_Y = df.to_numpy()[:,-1]

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
                    SGDRegressor(max_iter=100, verbose=0))
a = clf.fit(train_X, train_Y)

ress = zip(feature_names[1:-1], clf.named_steps['sgdregressor'].coef_)
for r in ress:
    #print(r)
    pass

df_test = pd.read_excel('test_timeseries.xlsx', index_col=None)
test_X = df_test.to_numpy()[:,1:-1]

for x, y in zip(company_names, a.predict(train_X)):
    print(x, y)

print('\"必翔106\" without change', a.predict(test_X)[0])
print("\"鴻友104\" without change", a.predict(test_X)[1])
exit()

for idx, point in enumerate(pca.transform(train_X[8:10])):
    plt.scatter(point[0], point[1], c=('b' if train_Y[idx] == 0 else 'r'))
    plt.annotate(company_names[idx+8], (point[0], point[1]), c='#000000' if train_Y[idx] == 0 else 'b', fontproperties=myfont)



# points = np.array(list(itertools.product(np.arange(-9, -4, 0.01), np.arange(74, 76, 0.01))))
# preds = a.predict(points)

# cs = []
# for p, pred in zip(points, preds):
#     weight = max(0, min(255, int(pred * 255)))
#     print('#{:02x} {:02x} {:02x}'.format(weight, weight, weight))
#     cs.append('#{:02x}{:02x}{:02x}'.format(weight, weight, weight))

# plt.scatter(points[:, 0], points[:, 1], s=1, c=cs, alpha=0.5)


plt.savefig('test.png')
