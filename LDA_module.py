# coding=utf-8
# created by czc on 2017.7.4

from nltk.tokenize import RegexpTokenizer           # 引入 nltk 的 分词模块
from stop_words import get_stop_words               # 引入 stopwords 里面的分词模块
from nltk.stem import WordNetLemmatizer             # 引入 nltk 的 词还原模块
from gensim import corpora, models, similarities
import matplotlib.pyplot as plt                     # 引入 pyplot，并将其别名为plt，为作图使用


class LDA_module:
    __doc_set = []                # 训练集
    __num_topics = 0              # 主题的个数
    __num_words = 0               # 主题包含的关键字个数
    __num_traversals = 20         # 遍历次数
    __train_set_dict = []         # 训练集字典
    __train_set_corpus = []       # 训练集词频
    __tfidf_model = []            # TF-IDF 模型
    __ldamodel = []               # 对应产生的 LDA 模型
    my_stoplist = []              # 增加特定应用的停用词表（本应用侧重医患对话）

    def __init__(self, doc_set, num_topics, num_words, num_traversals):
        print("初始化 ... ")
        self.doc_set = doc_set
        self.num_topics = num_topics
        self.num_words = num_words
        self.num_traversals = num_traversals
        f = open("G:\PyCharmWorkSpace/mystopword.txt", "r")
        content = f.readlines()                                 # 读取文本内容
        self.my_stoplist = set(content[0].split())              # 读取一行

    def trainmodel(self):
        print("开始训练 LDA 模型 ... ")

        tokenizer = RegexpTokenizer(r'\w+')                     # 创建一个去除标点符号等特殊字符的正则表达式分词器
        wnl = WordNetLemmatizer()                               # 创建 WordNetLemmatizer 用来做词还原
        en_stop = get_stop_words('en')                          # create English stop words list

        texts = []
        for i in self.doc_set:                                  # 对训练集做遍历
            # clean and tokenize document string
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)                    # 分词

            # remove stop words from tokens
            stopped_tokens1 = [i for i in tokens if not i in en_stop]                       # 移去常见的停用词
            stopped_tokens2 = [i for i in stopped_tokens1 if not i in self.my_stoplist]     # 移去针对应用设置的停用词

            # stem tokens
            stemmed_tokens = [wnl.lemmatize(i) for i in stopped_tokens2]

            # add tokens to list
            texts.append(stemmed_tokens)
            print(stemmed_tokens)

        # turn our tokenized documents into a id <-> term dictionary
        dictionary = corpora.Dictionary(texts)
        self.__train_set_dict = dictionary

        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in texts]
        self.__train_set_corpus = corpus

        tfidf = models.TfidfModel(corpus)       # 训练出 TF-IDF 模型
        self.__tfidf_model = tfidf
        corpus_tfidf = tfidf[corpus]            # 基于 TF-IDF 模型得到一个用tf-idf值表示的文档向量

        # generate LDA model
        ldamodel = models.ldamodel.LdaModel(corpus_tfidf, num_topics=self.num_topics, id2word=dictionary, passes=self.num_traversals)
        self.__ldamodel = ldamodel
        print("训练完成，主题模型：")
        print(ldamodel.print_topics(num_topics=self.num_topics, num_words=self.num_words))
        # print(ldamodel.print_topics(2))

    @staticmethod
    def show2dCorpora(corpus):
        nodes = list(corpus)
        ax0 = [x[0][1] for x in nodes]          # 绘制各个doc代表的点
        ax1 = [x[1][1] for x in nodes]
        plt.plot(ax0, ax1, 'o')
        plt.show()

    def predict(self, target_doc_set):
        print("开始预测 ... ")
        tokenizer = RegexpTokenizer(r'\w+')
        en_stop = get_stop_words('en')
        wnl = WordNetLemmatizer()

        # print(self.ldamodel.print_topics(num_topics=2, num_words=3))

        corpus_tfidf = self.__tfidf_model[self.__train_set_corpus]
        corpus_lda = self.__ldamodel[corpus_tfidf]
        # self.show2dCorpora(corpus_lda)

        corpus_simi_matrix = similarities.MatrixSimilarity(corpus_lda)

        count = 1
        for i in target_doc_set:
            print(count, ". 原文档：", i)

            # 提取词频向量
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)
            stopped_tokens1 = [i for i in tokens if not i in en_stop]
            stopped_tokens2 = [i for i in stopped_tokens1 if not i in self.my_stoplist]
            stemmed_tokens = [wnl.lemmatize(i) for i in stopped_tokens2]

            i_bow = self.__train_set_dict.doc2bow(stemmed_tokens)       # 把文档转换为稀疏矩阵
            print("    文档转换为稀疏矩阵：")
            print("   ", i_bow)
            i_lda = self.__ldamodel[i_bow]                              # 用之前训练好的 LDA 模型将其映射到二维的topic空间
            print("    文档通过 LDA 模型映射到", self.num_topics, "维的 topics 空间:")
            print("   ", i_lda)

            test_simi = corpus_simi_matrix[i_lda]                       # 计算余弦相似度（与训练集所有文档的相似度）
            print("    计算待测文档与各训练的余弦相似度：")
            print("   ", list(enumerate(test_simi)))

            count += 1


class sample:

    @staticmethod
    def run():
        doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
        doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
        doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
        doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
        doc_e = "Health professionals say that brocolli is good for your health."

        # compile sample documents into a list
        doc_set1 = [doc_a, doc_b, doc_c, doc_d, doc_e]
        doc_set2 = [doc_a, doc_b, doc_d]
        instance_ldamodule = LDA_module(doc_set2, num_topics=3, num_words=3, num_traversals=20)
        instance_ldamodule.trainmodel()
        instance_ldamodule.predict(doc_set1)
