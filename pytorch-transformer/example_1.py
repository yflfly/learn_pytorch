# coding:utf-8
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
print(classifier('We are very happy to show you the 🤗 Transformers library.'))
print(classifier('很开心看到你！'))

'''
利用transformer封装好的NLP相关的方法进行NLP任务的处理
上述实例进行NLP任务中  情感分析任务\

运行的结果如下所示：
Downloading: 100%|██████████| 230/230 [00:00<00:00, 57.2kB/s]
[{'label': 'POSITIVE', 'score': 0.9997795224189758}]
[{'label': 'POSITIVE', 'score': 0.8704843521118164}]
'''


# 例如：我在我的 dotvim 文件夹下一个 screenshots 目录，在该目录里有一个 vim-screenshot.jpg 截图。那么添加链接的方式如下
#
#  ![image](https://github.com/ButBueatiful/dotvim/raw/master/screenshots/vim-screenshot.jpg)