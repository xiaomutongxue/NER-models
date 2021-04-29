# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os


def load_file(file_path, limit_size=20):
    subs = []
    count = 0
    if not os.path.exists(file_path):
        return subs
    with open(file_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            i = line.strip()
            subs.append(i)
            if 0 < limit_size < count:
                break
            count += 1
    return subs


def horizontal_bio(predict_result_file, horizontal_file):
    with open(predict_result_file, 'r', encoding='utf-8') as fr, open(horizontal_file, 'w', encoding='utf-8') as fw:
        for line in fr:
            line = line.strip()
            terms = line.split("\t")
            i = terms[0]
            b = terms[1].split(',')[0]

            if not b:
                continue
            brand_idx = i.index(b)
            if brand_idx > -1:
                brand_start = brand_idx
                brand_len = len(b)
                if brand_len == 1:
                    continue
                out = ' '.join(['O'] * brand_start + ['B-ORG'] + ['I-ORG'] * (brand_len - 1) + ['O'] * (
                        len(i) - brand_start - brand_len))
            else:
                out = ' '.join(['O'] * len(i))
            fw.write(i + '\t' + out + '\n')
            print(i + '\t' + out)


def vertical_bio(horizontal_file, out_vertical_file):
    with open(horizontal_file, 'r', encoding='utf-8') as fr, open(out_vertical_file, 'w', encoding='utf-8') as fw:
        for line in fr:
            line = line.strip()
            terms = line.split('\t')
            chars = list(terms[0])
            tags = terms[1].split(' ')
            if len(chars) != len(tags):
                continue
            for i in range(len(chars)):
                fw.write(chars[i] + '\t' + tags[i] + '\n')
            fw.write('\n')


def main():
    # predict_file format: sentence '\t' brand1,brand2
    predict_file = 'sentence_brands.txt'
    horizontal_file = 'hor_train.txt'
    vertical_file = 'ver_train.txt'
    # to bio
    horizontal_bio(predict_file, horizontal_file)
    # to vertical bio
    vertical_bio(horizontal_file, vertical_file)


if __name__ == '__main__':
    main()
