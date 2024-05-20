'''
使用setfault进行简洁处理
'''

import sys
import re

WORD_RE = re.compile(r'\w+')

index = {}

with open(sys.argv[1], encoding='utf8') as fp:
    for line_no, line in enumerate(fp, 1):
        for match in WORD_RE.finditer(line):
            word = match.group()
            col_no = match.start()+1
            loc = (line_no, col_no)
            index.setdefault(word, []).append(loc) # 直接设定字典找不到的单词的value为[]即可

for word in sorted(index, key=str.upper):
    print(word, index[word])