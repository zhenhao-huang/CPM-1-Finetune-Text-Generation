import json
import os

os.makedirs('./data/lyric/preprocess', exist_ok=True)
os.makedirs('./data/lyric/preprocessed_id/train_dev_test_text', exist_ok=True)

# 预处理为列表文本，preprocess格式为[["一行歌词", ..., "一行歌词"], ["一行歌词", ..., "一行歌词"], ["一行歌词", ..., "一行歌词"]]
preprocess = []
sub_text = ''
with open('./data/lyric/preprocess/lyric.txt', 'r', encoding='utf-8') as f:
    text = f.readlines()
    for i in text:
        if i != '\n':
            sub_text += ''.join(i)
        elif sub_text != '':
            preprocess.append(sub_text)
            sub_text = ''
    preprocess.append(sub_text)
    preprocess = [i.strip().split('\n') for i in preprocess]
    print(len(preprocess))

# ---------------------------------train-dev-test---------------------------------------

# 输出为["一行歌词", ..., "一行歌词"]\n["一行歌词", ..., "一行歌词"]\n["一行歌词", ..., "一行歌词"]
with open('./data/lyric/preprocessed_id/train_dev_test_text/train.json', 'w', encoding='utf-8') as f:
    for i in preprocess[0:-100]:  # train
        preprocess_text = i
        a = json.dumps(preprocess_text, ensure_ascii=False)
        f.write(a + '\n')

with open('./data/lyric/preprocessed_id/train_dev_test_text/dev.json', 'w', encoding='utf-8') as f:
    for i in preprocess[-100:-50]:  # dev
        preprocess_text = i
        a = json.dumps(preprocess_text, ensure_ascii=False)
        f.write(a + '\n')

with open('./data/lyric/preprocessed_id/train_dev_test_text/test.json', 'w', encoding='utf-8') as f:
    for i in preprocess[-50:]:  # test
        preprocess_text = i
        a = json.dumps(preprocess_text, ensure_ascii=False)
        f.write(a + '\n')
