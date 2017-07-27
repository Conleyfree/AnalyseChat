# coding=utf-8

from nltk.tokenize import RegexpTokenizer       # 引入 nltk 的 分词模块
from stop_words import get_stop_words           # 引入 stopwords 里面的分词模块
from nltk.stem.porter import PorterStemmer      # 引入 PorterStemmer 模块作词干提取
from gensim import corpora, models
import gensim


# 导入文档
doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

# 匹配所有单字字符，直到其遇到像空格这样的非单字的字符，就划分出一个词
tokenizer = RegexpTokenizer(r'\w+')

# 转换为小写
raw_a = doc_a.lower()
raw_b = doc_a.lower()
raw_c = doc_c.lower()
raw_d = doc_d.lower()
raw_e = doc_e.lower()

# 对单个文档调用 tokenizer 进行分词
tokens_a = tokenizer.tokenize(raw_a)
tokens_b = tokenizer.tokenize(raw_b)
tokens_c = tokenizer.tokenize(raw_c)
tokens_d = tokenizer.tokenize(raw_d)
tokens_e = tokenizer.tokenize(raw_e)

# create English stop words list
en_stop = get_stop_words('en')

# remove stop words from tokens
stopped_tokens_a = [i for i in tokens_a if not i in en_stop]
stopped_tokens_b = [i for i in tokens_b if not i in en_stop]
stopped_tokens_c = [i for i in tokens_c if not i in en_stop]
stopped_tokens_d = [i for i in tokens_d if not i in en_stop]
stopped_tokens_e = [i for i in tokens_e if not i in en_stop]

print("移除停用词之前的分词结果：", tokens_a)
print("移除停用词之后：", stopped_tokens_a)

# Create p_stemmer of class PorterStemmer : Porter stemming algorithm 是使用最广泛的词干提取方法
p_stemmer = PorterStemmer()

# stem token
texts_a = [p_stemmer.stem(i) for i in stopped_tokens_a]
texts_b = [p_stemmer.stem(i) for i in stopped_tokens_b]
texts_c = [p_stemmer.stem(i) for i in stopped_tokens_c]
texts_d = [p_stemmer.stem(i) for i in stopped_tokens_d]
texts_e = [p_stemmer.stem(i) for i in stopped_tokens_e]

print("提取词干之后：", texts_a)

# 以上完成了导入文档，与清洗文档的三个步骤：分词、停用词、词干提取
# 清洗阶段的结果就是文本（texts），从单个的文档中整理出来的分好词，去除了停用词而且提取了词干的单词列表。

# 以下构建 document-term matrix
# document-term matrix 是一个描述文档词频的矩阵，每一行对应文档集中的一篇文档，每一列对应一个单词，
# 这个矩阵可以根据实际情况，采用不同的统计方法来构建。

texts = [texts_a, texts_b, texts_c, texts_d, texts_e]

# 遍历所有的文本，为每个不重复的单词分配一个单独的整数 ID，同时收集该单词出现次数以及相关的统计信息。
dictionary = corpora.Dictionary(texts)
print("输出所有单词及其id：", dictionary.token2id)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]
print(len(corpus))
print(corpus[0])        # 元组的形式是（单词 ID，词频）,输出词频矩阵的第 0 行（第一个文档 doc_a）

# generate LDA model
# num_topics: 必须。LDA 模型要求用户决定应该生成多少个主题。由于我们的文档集很小，所以我们只生成三个主题。
# id2word：必须。LdaModel 类要求我们之前的 dictionary 把 id 都映射成为字符串。
# passes：可选。模型遍历语料库的次数。遍历的次数越多，模型越精确。但是对于非常大的语料库，遍历太多次会花费很长的时间。
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)

# 以上，LDA 模型已经存储到 ldamodel 中，设置主题数与关键词个数，输出主题分析结果，
print(ldamodel.print_topics(num_topics=3, num_words=3))

# 调整主题数 和 遍历次数， 遍历次数越多，模型越精确
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=30)
print(ldamodel.print_topics(num_topics=2, num_words=4))




