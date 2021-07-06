import json
import os
import argparse
from tqdm import tqdm
from data_utils.tokenization_gpt2 import GPT2Tokenizer


def preprocess(data, tokenizer, split):
    text_ids = []
    for line in tqdm(data, desc="Preprocessing {}".format(split)):
        text = json.loads(line)  # 取出段落
        # max：1025，input_ids和labels为1024
        max_length = 1025
        for i in text:
            text_id = tokenizer.encode(i)
            # 使用滑动窗口截断，使每个text_id的长度不超过max_length
            text_split_id = [text_id[index:index + max_length] for index in range(0, len(text_id), max_length)]
            for j in text_split_id:
                text_ids.append(j)

    return text_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./data/novel/preprocessed_id/train_dev_test_text/', type=str, help="The input dir of original lyric data.")
    parser.add_argument("--tokenizer_path", type=str, help="The tokenizer path.", default="./bpe_3w_new")
    parser.add_argument("--output_dir", default='./data/novel/preprocessed_id', type=str, help="The processed data output dir.")

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
