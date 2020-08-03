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



![image](https://github.com/yflfly/learn_pytorch/tree/master/pytorch-transformer/image/example_1.png)