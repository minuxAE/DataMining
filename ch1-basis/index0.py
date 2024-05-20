'''
方案1代码
'''
import sys
import re

WORD_RE = re.compile(r'\w+') # 设计正则表达式，读取文本中的单词

# 将文本中出现的英文单词的位置（坐标）整理成字典
index = {} # 结果保存为字典，形式为{单词：[行号1，列号1]， [行号2，列号2]，...}

with open(sys.argv[1], encoding='utf8') as fp: # 读入文本（命令行输入文本）
    for line_no, line in enumerate(fp, 1): # 迭代每行信息
        for match in WORD_RE.finditer(line): # 正则匹配
            word = match.group() # 匹配到单词的信息
            colume_no = 1+match.start() # 列号
            loc = (line_no, colume_no) # 单词坐标信息保存为tuple

            occr = index.get(word, []) # 如果当前字典没有该单词，则设定为[]，否则将已有信息读取出来
            occr.append(loc) # 添加新的坐标
            index[word]=occr # 写回字典

for word in sorted(index, key=str.upper):
    print(word, index[word])