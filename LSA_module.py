# coding=utf-8
# created by czc on 2017.7.25

from nltk.tokenize import RegexpTokenizer           # 引入 nltk 的 分词模块
from stop_words import get_stop_words               # 引入 stopwords 里面的分词模块
from nltk.stem.porter import PorterStemmer          # 引入 PorterStemmer 模块作词干提取
from gensim import corpora, models, similarities
import matplotlib.pyplot as plt                     # 引入 pyplot，并将其别名为plt，为作图使用


class LSA_module:
    __doc_set = []                # 训练集
    __num_topics = 0              # 主题的个数
    __num_words = 0               # 主题包含的关键字个数
    __num_traversals = 20         # 遍历次数
    __train_set_dict = []         # 训练集字典
    __train_set_corpus = []       # 训练集词频
    __tfidf_model = []            # TF-IDF 模型
    __lsamodel = ''               # 对应产生的 LSA 模型

    def __init__(self, doc_set, num_topics, num_words, num_traversals):
        print("初始化 ... ")
        self.doc_set = doc_set
        self.num_topics = num_topics
        self.num_words = num_words
        self.num_traversals = num_traversals

    def trainmodel(self):
        print("开始训练 LSA（LSI） 模型 ... ")
        # 匹配所有单字字符，直到其遇到像空格这样的非单字的字符，就划分出一个词
        tokenizer = RegexpTokenizer(r'\w+')

        # create English stop words list
        en_stop = get_stop_words('en')

        # Create p_stemmer of class PorterStemmer
        p_stemmer = PorterStemmer()

        # list for tokenized documents in loop
        texts = []

        # loop through document list
        for i in self.doc_set:
            # clean and tokenize document string
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)

            # remove stop words from tokens
            stopped_tokens = [i for i in tokens if not i in en_stop]

            # stem tokens
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

            # add tokens to list
            texts.append(stemmed_tokens)

        # turn our tokenized documents into a id <-> term dictionary
        dictionary = corpora.Dictionary(texts)
        self.__train_set_dict = dictionary

        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in texts]
        self.__train_set_corpus = corpus

        tfidf = models.TfidfModel(corpus)       # 训练出 TF-IDF 模型
        self.__tfidf_model = tfidf
        corpus_tfidf = tfidf[corpus]            # 基于 TF-IDF 模型得到一个用tf-idf值表示的文档向量

        # generate LSA model
        lsimodel = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=self.num_topics)
        self.__lsamodel = lsimodel
        print("训练完成，LSA（LSI） 主题模型：(词前的数字代表权重)")
        print(lsimodel.print_topics(num_topics=self.num_topics, num_words=self.num_words))

    @staticmethod
    def show2dCorpora(corpus):
        nodes = list(corpus)
        ax0 = [x[0][1] for x in nodes]  # 绘制各个doc代表的点
        ax1 = [x[1][1] for x in nodes]
        plt.plot(ax0, ax1, 'o')
        plt.show()

    def predict(self, target_doc_set):
        print("开始预测 ... ")
        tokenizer = RegexpTokenizer(r'\w+')
        en_stop = get_stop_words('en')
        p_stemmer = PorterStemmer()

        corpus_tfidf = self.__tfidf_model[self.__train_set_corpus]
        corpus_lsa = self.__lsamodel[corpus_tfidf]
        # self.show2dCorpora(corpus_lsa)

        corpus_simi_matrix = similarities.MatrixSimilarity(corpus_lsa)

        count = 1
        for i in target_doc_set:
            print(count, ". 原文档：", i)

            # 提取词频向量
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)
            stopped_tokens = [i for i in tokens if not i in en_stop]
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

            i_bow = self.__train_set_dict.doc2bow(stemmed_tokens)       # 把文档转换为稀疏矩阵
            print("    文档转换为稀疏矩阵：")
            print("   ", i_bow)
            i_lsa = self.__lsamodel[i_bow]                              # 用之前训练好的 LSA 模型将其映射到二维的topic空间
            print("    文档通过 LSA 模型映射到", self.num_topics, "维 topics 空间:")
            print("   ", i_lsa)

            test_simi = corpus_simi_matrix[i_lsa]                     # 计算余弦相似度（与训练集所有文档的相似度）
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
        instance_lsamodule = LSA_module(doc_set2, num_topics=3, num_words=3, num_traversals=20)
        instance_lsamodule.trainmodel()
        instance_lsamodule.predict(doc_set1)