import json
import glob
import os

os.makedirs('./data/novel/preprocess', exist_ok=True)
os.makedirs('./data/novel/preprocessed_id/train_dev_test_text', exist_ok=True)

# 清洗数据
# symbol = ["。", "”", "：", "…", "？", "！", "）", "—", ".", "?", "，", "\"", "；", "．", ":", "", "、", "“", "’", "!", "】",
#           ")", ",", ";", "》"]
# file = glob.glob("./data/novel/preprocess/*.txt")
# for i in file:
#     with open(i, 'r', encoding='utf-8-sig') as f:
#         text = f.readlines()
#         # print(text)
#         a = {i + 1: j.replace('\n', '')[-1] for i, j in enumerate(text) if
#              j.replace('\n', '') != '' and j.replace('\n', '')[-1] not in symbol}
#         print(i, '\n', a)

# 预处理为列表文本，preprocess格式为[["段落"], ["段落"], ["段落"]]
preprocess = []
files = glob.glob("./data/novel/preprocess/*.txt")
for file in files:
    sub_text = ""
    with open(file, 'r', encoding='utf-8-sig') as f:
        text = f.readlines()
    for i in text:
        if i != '\n':
            sub_text += i
        elif sub_text != "":
            preprocess.append(sub_text)  # 将每个段落添加到列表里
            sub_text = ""
    preprocess.append(sub_text)  # 将每个段落添加到列表里
preprocess = [[i.replace('\n', '')] for i in preprocess]  # 去掉换行符
# print(preprocess)
print(len(preprocess))

# ---------------------------------train-dev-test---------------------------------------

# 输出为["段落"]\n["段落"]\n["段落"]
with open('./data/novel/preprocessed_id/train_dev_test_text/train.json', 'w', encoding='utf-8') as f:
    for i in preprocess[0:-10]:  # train
        preprocess_text = i
        a = json.dumps(preprocess_text, ensure_ascii=False)
        f.write(a + '\n')

with open('./data/novel/preprocessed_id/train_dev_test_text/dev.json', 'w', encoding='utf-8') as f:
    for i in preprocess[-10:-5]:  # dev
        preprocess_text = i
        a = json.dumps(preprocess_text, ensure_ascii=False)
        f.write(a + '\n')

with open('./data/novel/preprocessed_id/train_dev_test_text/test.json', 'w', encoding='utf-8') as f:
    for i in preprocess[-5:]:  # test
        preprocess_text = i
        a = json.dumps(preprocess_text, ensure_ascii=False)
        f.write(a + '\n')
