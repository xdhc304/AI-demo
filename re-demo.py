# encoding: UTF-8
import re
 
# 将正则表达式编译成Pattern对象
pattern = re.compile(r'hello.*\!')
 
# 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
match = pattern.match('hello, xdhc304! How are you?')
 
if match:
  # 使用Match获得分组信息
  print(match.group())

regex_1 = re.compile(r"""\d +  # 数字部分
                         \.    # 小数点部分
                         \d *  # 小数的数字部分""", re.X)
regex_2 = re.compile(r"\d+\.\d*")

m = re.match(r'(\w+) (\w+)(?P<sign>.*)', 'hello xdhc304!')
 
print("m.string:", m.string)
print("m.re:", m.re)
print("m.pos:", m.pos)
print("m.endpos:", m.endpos)
print("m.lastindex:", m.lastindex)
print("m.lastgroup:", m.lastgroup)
 
print("m.group(1,2):", m.group(1, 2))
print("m.groups():", m.groups())
print("m.groupdict():", m.groupdict())
print("m.start(2):", m.start(2))
print("m.end(2):", m.end(2))
print("m.span(2):", m.span(2))
print(r"m.expand(r'\2 \1\3'):", m.expand(r'\2 \1\3'))

p = re.compile(r'(\w+) (\w+)(?P<sign>.*)', re.DOTALL)
 
print("p.pattern:", p.pattern)
print("p.flags:", p.flags)
print("p.groups:", p.groups)
print("p.groupindex:", p.groupindex)

# 将正则表达式编译成Pattern对象 
pattern = re.compile(r'H.*g')
 
# 使用search()查找匹配的子串，不存在能匹配的子串时将返回None 
# 这个例子中使用match()无法成功匹配 
match = pattern.search('hello xdhc304!')
 
if match: 
  # 使用Match获得分组信息 
  print(match.group()) 
 
p = re.compile(r'\d+')
print(p.split('one1two2three3four4'))

p = re.compile(r'\d+')
print(p.findall('one1two2three3four4'))
 
p = re.compile(r'\d+')
for m in p.finditer('one1two2three3four4'):
  print(m.group())

p = re.compile(r'(\w+) (\w+)')
s = 'i say, hello xdhc304!'
 
print(p.sub(r'\2 \1', s))
 
def func(m):
  return m.group(1).title() + ' ' + m.group(2).title()
 
print(p.sub(func, s))

p = re.compile(r'(\w+) (\w+)')
s = 'i say, hello xdhc304!'
 
print(p.subn(r'\2 \1', s))
 
def func(m):
  return m.group(1).title() + ' ' + m.group(2).title()
 
print(p.subn(func, s))