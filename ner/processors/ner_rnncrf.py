# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import json
import os

import numpy as np
import torch

FILE_VOCAB = "vocab.json"
FILE_TAGS = "labels.json"
FILE_DATASET_CACHE = "dataset_cache.npz"
FILE_ARGUMENTS = "args.json"
FILE_MODEL = "bilstm_crf_model.pth"
FILE_PREDICT = "predicted.json"

START_TAG = "<START>"
STOP_TAG = "<STOP>"

PAD = "<PAD>"
OOV = "<OOV>"


def save_json_file(obj, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=4))


def load_json_file(file_path):
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


class Processor:
    def __init__(self, output_dir, data_dir='', verbose=True):
        self.output_dir = output_dir
        self.verbose = verbose
        if data_dir:
            self._save_vocab(data_dir, output_dir)
        self.vocab, self.vocab_dict = self._load_list_file(FILE_VOCAB, offset=1, verbose=verbose)
        self.tags, self.tags_dict = self._load_list_file(FILE_TAGS, verbose=verbose)

        self.PAD_IDX = 0
        self.OOV_IDX = len(self.vocab)
        self._adjust_vocab()

    def _read_text(self, input_file):
        lines = []
        with open(input_file, 'r', encoding='utf-8') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n" or line == "end":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split()
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines

    def _load_list_file(self, file_name, offset=0, verbose=False):
        file_path = os.path.join(self.output_dir, file_name)
        if not os.path.exists(file_path):
            raise ValueError('"{}" file does not exist.'.format(file_path))
        else:
            elements = load_json_file(file_path)
            elements_dict = {w: idx + offset for idx, w in enumerate(elements)}
            if verbose:
                print("file {} loaded".format(file_path))
            return elements, elements_dict

    def _adjust_vocab(self):
        self.vocab.insert(0, PAD)
        self.vocab_dict[PAD] = 0

        self.vocab.append(OOV)
        self.vocab_dict[OOV] = len(self.vocab) - 1

    def _save_vocab(self, data_dir, dst_dir):
        vocab_file = os.path.join(dst_dir, FILE_VOCAB)
        tag_file = os.path.join(dst_dir, FILE_TAGS)

        if not os.path.exists(vocab_file):
            vocab_set = set()
            tag_set = set()
            for i in os.listdir(data_dir):
                data_path = os.path.join(data_dir, i)
                if os.path.exists(data_path):
                    lines = self._read_text(data_path)
                    for (i, line) in enumerate(lines):
                        text_a = line['words']
                        # BIOS
                        labels = []
                        for x in line['labels']:
                            if 'M-' in x:
                                labels.append(x.replace('M-', 'I-'))
                            elif 'E-' in x:
                                labels.append(x.replace('E-', 'I-'))
                            else:
                                labels.append(x)
                        vocab_set |= set(text_a)
                        tag_set |= set(labels)
            save_json_file(list(vocab_set), vocab_file)
            save_json_file(list(tag_set), tag_file)

    def load_dataset(self, data_dir, output_dir, val_split, test_split, max_seq_len):
        """load the train set
        :return: (xs, ys)
            xs: [B, L]
            ys: [B, L, C]
        """
        cache_file = os.path.join(output_dir, FILE_DATASET_CACHE)
        if not os.path.exists(cache_file):
            xs, ys = self._build_corpus(data_dir, max_seq_len)
            # save train set
            np.savez(cache_file, xs=xs, ys=ys)
            print("dataset cache({}, {}) => {}".format(xs.shape, ys.shape, cache_file))

        else:
            print("loading dataset {} ...".format(cache_file))
            dataset = np.load(cache_file)
            xs, ys = dataset["xs"], dataset["ys"]

        xs, ys = map(torch.tensor, (xs, ys))

        # split the dataset
        total_count = len(xs)
        assert total_count == len(ys)
        val_count = int(total_count * val_split)
        test_count = int(total_count * test_split)
        train_count = total_count - val_count - test_count
        assert train_count > 0 and val_count > 0

        indices = np.cumsum([0, train_count, val_count, test_count])
        datasets = [(xs[s:e], ys[s:e]) for s, e in zip(indices[:-1], indices[1:])]
        print("datasets loaded:")
        for (xs_, ys_), name in zip(datasets, ["train", "val", "test"]):
            print("\t{}: {}, {}".format(name, xs_.shape, ys_.shape))
        return datasets

    def decode_tags(self, batch_tags):
        batch_tags = [
            [self.tags[t] for t in tags]
            for tags in batch_tags
        ]
        return batch_tags

    def sent_to_vector(self, sentence, max_seq_len=0):
        max_seq_len = max_seq_len if max_seq_len > 0 else len(sentence)
        vec = [self.vocab_dict.get(c, self.OOV_IDX) for c in sentence[:max_seq_len]]
        return vec + [self.PAD_IDX] * (max_seq_len - len(vec))

    def tags_to_vector(self, tags, max_seq_len=0):
        max_seq_len = max_seq_len if max_seq_len > 0 else len(tags)
        vec = [self.tags_dict[c] for c in tags[:max_seq_len]]
        return vec + [0] * (max_seq_len - len(vec))

    def _build_corpus(self, data_dir, max_seq_len):
        xs, ys = [], []
        for i in os.listdir(data_dir):
            data_path = os.path.join(data_dir, i)
            if os.path.exists(data_path):
                lines = self._read_text(data_path)
                for (i, line) in enumerate(lines):
                    text_a = line['words']
                    # BIOS
                    labels = []
                    for x in line['labels']:
                        if 'M-' in x:
                            labels.append(x.replace('M-', 'I-'))
                        elif 'E-' in x:
                            labels.append(x.replace('E-', 'I-'))
                        else:
                            labels.append(x)
                    xs.append(self.sent_to_vector(text_a, max_seq_len=max_seq_len))
                    ys.append(self.tags_to_vector(labels, max_seq_len=max_seq_len))
                    if len(text_a) != len(labels):
                        raise ValueError('"sentence length({})" != "tags length({})" in line {}"'.format(
                            len(text_a), len(labels), i + 1))

        xs, ys = np.asarray(xs), np.asarray(ys)
        return xs, ys
