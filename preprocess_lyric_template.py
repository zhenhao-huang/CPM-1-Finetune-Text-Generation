import json
import os
import argparse
from tqdm import tqdm
from data_utils.tokenization_gpt2 import GPT2Tokenizer


def preprocess(data, tokenizer, split):
    text_ids = []
    for line in tqdm(data, desc="Preprocessing {}".format(split)):
        text = json.loads(line)
        text_id = []
        # max：1025，input_ids和labels为1024
        max_length = 1025
        eod_id = tokenizer.encoder["<eod>"]
        for i in text:
            length = len(text_id) + len(tokenizer.encode(i)) + len([eod_id])
            # 转id之前判断转id之后的文本长度是否超过max_length
            if length <= max_length:
                text_id.extend(tokenizer.encode(i))
                text_id.append(eod_id)
            else:
                # 若超过max_length则将上一次序列加入text_ids，text_id重置为[]
                text_ids.append(text_id)
                # 若某些单行文本转id之后长度超过max_length，则打印出来根据实际情况手动换行
                if len(text_id) > max_length:
                    print('\n', len(text_id), i)
                text_id = []
                text_id.extend(tokenizer.encode(i))
                text_id.append(eod_id)
        text_ids.append(text_id)
        if len(text_id) > max_length:
            print('\n', len(text_id), i)

    return text_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./data/lyric/preprocessed_id/train_dev_test_text/', type=str,
                        help="The input dir of original lyric data.")
    parser.add_argument("--tokenizer_path", type=str, help="The tokenizer path.", default="./bpe_3w_new")
    parser.add_argument("--output_dir", default='./data/lyric/preprocessed_id', type=str,
                        help="The processed data output dir.")

    args = parser.parse_args()
    tokenizer = GPT2Tokenizer(os.path.join(args.tokenizer_path, 'vocab.json'),
                              os.path.join(args.tokenizer_path, 'chinese_vocab.model'))
    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["train", "dev", "test"]:
        with open(os.path.join(args.data_dir, "{}.json".format(split)), "r") as f:
            data = f.readlines()
        text_ids = preprocess(data, tokenizer, split)

        # 输出为[token_id, token_id, token_id]\n[token_id, token_id, token_id]\n[token_id, token_id, token_id]
        with open(os.path.join(args.output_dir, "{}.json".format(split)), "w") as f:
            for i in text_ids:
                preprocess_text = i
                a = json.dumps(preprocess_text, ensure_ascii=False)
                f.write(a + '\n')
