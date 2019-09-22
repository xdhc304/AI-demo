import operator

s = ' hello, world!'
print(s.strip())
print(s.lstrip(' hello, '))
print(s.rstrip('!'))

sStr1 = 'strcat'
sStr2 = 'append'
sStr1 += sStr2
print(sStr1)

# < 0 为未找到
sStr1 = 'strchr'
sStr2 = 'r'
nPos = sStr1.index(sStr2)
print(nPos)

sStr1 = 'strchr'
sStr2 = 'strch'
print(operator.lt(sStr2,sStr1))
print(operator.le(sStr2,sStr1))
print(operator.eq(sStr1,sStr2))
print(operator.ne(sStr1,sStr2))
print(operator.gt(sStr1,sStr1))
print(operator.ge(sStr1,sStr1))

# lt(a,b)相当于a<b 从第一个数字或字母（ASCII）比大小 
# le(a,b)相当于a<=b
# eq(a,b)相当于a==b 字母完全一样，返回True,
# ne(a,b)相当于a!=b
# gt(a,b)相当于a>b
# ge(a,b)相当于a>=b

sStr1 = 'JCstrlwr'
sStr1 = sStr1.upper()
#sStr1 = sStr1.lower()
print(sStr1)

sStr1 = 'abcdefg'
sStr1 = sStr1[::-1]
print(sStr1)

sStr1 = 'abcdefg'
sStr2 = 'cde'
print(sStr1.find(sStr2))

sStr1 = 'ab,cde,fgh,ijk'
sStr2 = ','
sStr1 = sStr1[sStr1.find(sStr2) + 1:]
print(sStr1)
#或者
s = 'ab,cde,fgh,ijk'
print(s.split(','))

#version 1
import re
from collections import Counter

def get_max_value_v1(text):
  text = text.lower()
  result = re.findall('[a-zA-Z]', text)  # 去掉列表中的符号符
  count = Counter(result)  # Counter({'l': 3, 'o': 2, 'd': 1, 'h': 1, 'r': 1, 'e': 1, 'w': 1})
  count_list = list(count.values())
  max_value = max(count_list)
  max_list = []
  for k, v in count.items():
    if v == max_value:
      max_list.append(k)
  max_list = sorted(max_list)
  return max_list[0]

#version 2
from collections import Counter

def get_max_value(text):
  count = Counter([x for x in text.lower() if x.isalpha()])
  m = max(count.values())
  return sorted([x for (x, y) in count.items() if y == m])[0]

#version 3
import string

def get_max_value(text):
  text = text.lower()
  return max(string.ascii_lowercase, key=text.count)

max(range(6), key = lambda x : x>2)
# >>> 3
# 带入key函数中，各个元素返回布尔值，相当于[False, False, False, True, True, True]
# key函数要求返回值为True，有多个符合的值，则挑选第一个。

max([3,5,2,1,4,3,0], key = lambda x : x)
# >>> 5
# 带入key函数中，各个元素返回自身的值，最大的值为5，返回5.

max('ah', 'bf', key=lambda x: x[1])
# >>> 'ah'
# 带入key函数，各个字符串返回最后一个字符，其中'ah'的h要大于'bf'中的f，因此返回'ah'

max('ah', 'bf', key=lambda x: x[0])
# >>> 'bf'
# 带入key函数，各个字符串返回第一个字符，其中'bf'的b要大于'ah'中的a，因此返回'bf'

text = 'Hello World'
max('abcdefghijklmnopqrstuvwxyz', key=text.count)
# >>> 'l'
# 带入key函数，返回各个字符在'Hello World'中出现的次数，出现次数最多的字符为'l',因此输出'l'

#T  h  e     M  i  s  s  i  s  s  i  p  p  i     R  i  v  e  r
#[1, 1, 2, 2, 1, 5, 4, 4, 5, 4, 4, 5, 2, 2, 5, 2, 1, 5, 1, 2, 1]
sentence='The Mississippi River'

def count_chars(s):
  s=s.lower()
  count=list(map(s.count,s))
  return (max(count))

print(count_chars(sentence))