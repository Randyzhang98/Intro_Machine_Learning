import pandas as pd
#第二参数代表要读取的sheet, 0表示第一个, 1表示第二个..., pandas默认读取第一个
df = pd.read_csv('./data_sets/data_batch1.csv')
print ('read_success')
df = df.iloc[:5000]
import seaborn as sns
import matplotlib.pyplot as plt 
# #初始化
# sns.set(style='whitegrid')
# #绘制分布矩阵
# sns.pairplot(df,palette = 'husl',diag_kind="kde",plot_kws=dict(s=1, edgecolor="r", linewidth=1))
# #保存图片
# plt.savefig('ccpp_UNSW.png')
# plt.show()
correlation = df.corr()
print(correlation)
sns.heatmap(correlation)
plt.savefig('correlation_UNSW.png')
plt.show()