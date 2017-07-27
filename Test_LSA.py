# coding=utf-8
# created by czc on 2017.7.15

from numpy import zeros
from scipy.linalg import svd
import matplotlib.pyplot as plt         # 导入pyplot，并将其别名为plt，为作图使用


class LSA(object):

    stopwords = ['and', 'edition', 'for', 'in', 'little', 'of', 'the', 'to']
    ignorechars = ",:'!"
    keys = []       # 单词集合
    A = []          # 用来保存 单词-文章（标题）矩阵

    # Define LSA Class
    def __init__(self, stopwords, ignorechars):
        self.stopwords = stopwords              # “停止词” 集合
        self.ignorechars = ignorechars          # 忽略的字符
        self.wdict = {}                         # 字典，记录某个单词出现的文本集合
        self.dcount = 0                         # 记录文档数

    # Parse Documents
    def parse(self, doc):
        words = doc.split()                                     # 文档截取成单词集合

        # 引用 maketrans 函数。
        outtab = "    "
        trantab = str.maketrans(self.ignorechars, outtab)
        for w in words:
            w = w.lower().translate(trantab).rstrip()           # 过滤掉 self.ignorechars 中的单词，替换为空
            print(w)
            if w in self.stopwords:
                continue
            elif w in self.wdict:                               # if 单词 w 已经被记录在字典 wdict 中
                self.wdict[w].append(self.dcount)               # 字典中单词 w 对应的单词出现文档列表中，新增当前的文档序号
            else:
                self.wdict[w] = [self.dcount]
        self.dcount += 1

    # Build the Count Matrix
    def build(self):
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]        # 只提取在两个以上文档中出现的单词
        self.keys.sort()
        self.A = zeros([len(self.keys), self.dcount])                               # 创建 单词-标题（文章）矩阵
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.A[i, d] += 1

    def printA(self):
        print(self.A)

        # 图形化输出
        u, s, vt = svd(self.A)          # 调用 SVD 算法进行矩阵分析，做降维
        print('''\r每个单词在语义空间中的坐标为：''')
        print(u)
        print("""\r词-标题矩阵包含的语义空间的有效维度""")
        print(s)
        print("""\r每个文章在语义空间中的坐标为""")
        print(vt)
        print("""\r""")

        plt.title("LSA")                # 设置图表的标题
        plt.xlabel(u'dimention2')       # 注释 x 坐标轴
        plt.ylabel(u'dimention3')       # 注释 y 坐标轴

        title_tags = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']
        vdemention2 = vt[1]
        vdemention3 = vt[2]
        for j in range(len(vdemention2)):
            plt.text(vdemention2[j], vdemention3[j], title_tags[j])
        plt.plot(vdemention2, vdemention3, '.')

        ut = u.T
        demention2 = ut[1]
        demention3 = ut[2]
        for i in range(len(demention2)):
            plt.text(demention2[i], demention3[i], self.keys[i])
        plt.plot(demention2, demention3, '.')

        plt.show()


titles = [
        "The Neatest Little Guide to Stock Market Investing",
        "Investing For Dummies, 4th Edition",
        "The Little Book of Common Sense Investing: The Only Way to Guarantee Your Fair Share of Stock Market Returns",
        "The Little Book of Value Investing",
        "Value Investing: From Graham to Buffett and Beyond",
        "Rich Dad's Guide to Investing: What the Rich Invest in, That the Poor and the Middle Class Do Not!",
        "Investing in Real Estate, 5th Edition",
        "Stock Investing For Dummies",
        "Rich Dad's Advisors: The ABC's of Real Estate Investing: The Secrets of Finding Hidden Profits Most Investors Miss"
    ]

stopword = ['and', 'edition', 'for', 'in', 'little', 'of', 'the', 'to']
ignorechar = ",:'!"
mylsa = LSA(stopword, ignorechar)
for t in titles:
    mylsa.parse(t)
mylsa.build()
mylsa.printA()