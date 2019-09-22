# encoding=utf-8
from __future__ import unicode_literals
import jieba

seg_list = jieba.cut("我在学习自然语言处理", cut_all=True)
print(seg_list)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我在学习自然语言处理", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他毕业于上海交通大学，在百度深度学习研究院进行研究")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在哈佛大学深造")  # 搜索引擎模式
print(", ".join(seg_list))

result_lcut = jieba.lcut("小明硕士毕业于中国科学院计算所，后在哈佛大学深造")
print(result_lcut)
print(" ".join(result_lcut))
print(" ".join(jieba.lcut_for_search("小明硕士毕业于中国科学院计算所，后在哈佛大学深造")))

print('/'.join(jieba.cut('如果放到旧字典中将出错。', HMM=False)))
jieba.suggest_freq(('中', '将'), True)
print('/'.join(jieba.cut('如果放到旧字典中将出错。', HMM=False)))

import jieba.analyse as analyse
lines = open('NBA.txt', 'rb').read()
print("  ".join(analyse.extract_tags(lines, topK=20, withWeight=False, allowPOS=())))

lines = open(u'西游记.txt', 'rb').read()
print("  ".join(analyse.extract_tags(lines, topK=20, withWeight=False, allowPOS=())))

import jieba.analyse as analyse
lines = open('NBA.txt', 'rb').read()
print("  ".join(analyse.textrank(lines, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))))
print("---------------------我是分割线----------------")
print("  ".join(analyse.textrank(lines, topK=20, withWeight=False, allowPOS=('ns', 'n'))))

lines = open(u'西游记.txt', 'rb').read()
print("  ".join(analyse.textrank(lines, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))))

import jieba.posseg as pseg
words = pseg.cut("我爱自然语言处理")
for word, flag in words:
  print('%s %s' % (word, flag))

# import sys
# import time
# import jieba

# jieba.enable_parallel()
# content = open(u'西游记.txt',"r").read()
# t1 = time.time()
# words = "/ ".join(jieba.cut(content))
# t2 = time.time()
# tm_cost = t2-t1
# print('并行分词速度为 %s bytes/second' % (len(content)/tm_cost))

# jieba.disable_parallel()
# content = open(u'西游记.txt',"r").read()
# t1 = time.time()
# words = "/ ".join(jieba.cut(content))
# t2 = time.time()
# tm_cost = t2-t1
# print('非并行分词速度为 %s bytes/second' % (len(content)/tm_cost))

print("这是默认模式的tokenize")
result = jieba.tokenize(u'自然语言处理非常有用')
for tk in result:
    print("%s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))

print("\n-----------我是神奇的分割线------------\n")

print("这是搜索模式的tokenize")
result = jieba.tokenize(u'自然语言处理非常有用', mode='search')
for tk in result:
  print("%s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))

# -*- coding: UTF-8 -*-
# from __future__ import unicode_literals
import sys, os
sys.path.append("../")
from whoosh.index import create_in,open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser

analyzer = jieba.analyse.ChineseAnalyzer()
schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT(stored=True, analyzer=analyzer))
    
if not os.path.exists("tmp"):
  os.mkdir("tmp")

ix = create_in("tmp", schema) # for create new index
#ix = open_dir("tmp") # for read only
writer = ix.writer()

writer.add_document(
  title="document1",
  path="/a",
  content="This is the first document we’ve added!"
)

writer.add_document(
  title="document2",
  path="/b",
  content="The second one 你 中文测试中文 is even more interesting! 吃水果"
)

writer.add_document(
  title="document3",
  path="/c",
  content="买水果然后来世博园。"
)

writer.add_document(
  title="document4",
  path="/c",
  content="工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"
)

writer.add_document(
  title="document4",
  path="/c",
  content="咱俩交换一下吧。"
)

writer.commit()
searcher = ix.searcher()
parser = QueryParser("content", schema=ix.schema)

for keyword in ("水果世博园","你","first","中文","交换机","交换"):
  print(keyword+"的结果为如下：")
  q = parser.parse(keyword)
  results = searcher.search(q)
  for hit in results:
      print(hit.highlights("content"))
  print("\n--------------我是神奇的分割线--------------\n")

for t in analyzer("我的好朋友是李明;我爱北京天安门;IBM和Microsoft; I have a dream. this is intetesting and interested me a lot"):
  print(t.text)