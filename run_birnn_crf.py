# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from ner.models.birnncrf.birnn_crf import BiRnnCrf
from ner.processors.ner_rnncrf import Processor, load_json_file, FILE_MODEL, FILE_ARGUMENTS, save_json_file, \
    FILE_PREDICT
from ner.processors.ner_seq import ner_processors as processors
from ner.tools.common import logger, init_logger, seed_everything


def build_model(args, processor, load=True):
    model = BiRnnCrf(len(processor.vocab),
                     len(processor.tags),
                     embedding_dim=args.embedding_dim,
                     hidden_dim=args.hidden_dim,
                     num_rnn_layers=args.num_rnn_layers)

    # weights
    model_path = os.path.join(args.output_dir, FILE_MODEL)
    if os.path.exists(model_path) and load:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        logger.info("load model weights from {}".format(model_path))
    return model


def running_device(device):
    return device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _eval_model(model, device, dataloader, desc):
    model.eval()
    with torch.no_grad():
        # eval
        losses, nums = zip(*[
            (model.loss(xb.to(device), yb.to(device)), len(xb))
            for xb, yb in tqdm(dataloader, desc=desc)])
        return np.sum(np.multiply(losses, nums)) / np.sum(nums)


def _save_loss(losses, file_path):
    pd.DataFrame(data=losses, columns=["epoch", "batch", "train_loss", "val_loss"]).to_csv(file_path, index=False)


def _save_model(output_dir, model):
    model_path = os.path.join(output_dir, FILE_MODEL)
    torch.save(model.state_dict(), model_path)
    logger.info("save model => {}".format(model_path))


class WordsTagger:
    def __init__(self, output_dir, device=None):
        args_ = load_json_file(os.path.join(output_dir, FILE_ARGUMENTS))
        args = argparse.Namespace(**args_)
        args.output_dir = output_dir

        self.preprocessor = Processor(output_dir=output_dir, verbose=False)
        self.model = build_model(args, self.preprocessor, load=True)
        self.device = running_device(device)
        self.model.to(self.device)

        self.model.eval()

    def __call__(self, sentences):
        """predict texts
        :param sentences: a text or a list of text
        :return:
        """
        if not isinstance(sentences, (list, tuple)):
            raise ValueError("sentences must be a list of sentence")

        try:
            sent_tensor = np.asarray([self.preprocessor.sent_to_vector(s) for s in sentences])
            sent_tensor = torch.from_numpy(sent_tensor).to(self.device)
            with torch.no_grad():
                _, tags = self.model(sent_tensor)
            tags = self.preprocessor.decode_tags(tags)
        except RuntimeError as e:
            logger.info("*** runtime error: {}".format(e))
            raise e
        return tags, self.tokens_from_tags(sentences, tags)

    @staticmethod
    def tokens_from_tags(sentences, tags_list):
        """extract entities from tags
        :param sentences: a list of sentence
        :param tags_list: a list of tags
        :return:
        """
        if not tags_list:
            return []

        def _tokens(sentence, ts):
            idx_entity = get_entity_bios(ts)
            tokens = []
            for entity in idx_entity:
                begin = entity[1]
                end = entity[2]
                tag = entity[0]
                tokens.append((sentence[begin:end+1],tag))
            return tokens

        tokens_list = [_tokens(sentence, ts) for sentence, ts in zip(sentences, tags_list)]
        return tokens_list


def get_entity_bios(seq, id2label=None):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
        id2label: id to label dict
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def predict(args):
    logger.info('predict sentence:')
    logger.info(str(args.predict_sentence))
    results = WordsTagger(args.output_dir, args.device)([args.predict_sentence])
    logger.info(json.dumps(results, ensure_ascii=False))
    save_json_file(results, os.path.join(args.output_dir, FILE_PREDICT))


def train(args):
    logger.info(str(args))
    save_json_file(vars(args), os.path.join(args.output_dir, FILE_ARGUMENTS))
    processor = Processor(output_dir=args.output_dir, data_dir=args.data_dir, verbose=True)
    model = build_model(args, processor, load=args.recovery)

    # loss
    loss_path = os.path.join(args.output_dir, "loss.csv")
    losses = pd.read_csv(loss_path).values.tolist() if args.recovery and os.path.exists(loss_path) else []

    # datasets
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = processor.load_dataset(
        args.data_dir, args.output_dir, args.val_split, args.test_split, max_seq_len=args.max_seq_len)
    train_dl = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    valid_dl = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    device = running_device(args.device)
    logger.info(device)
    model.to(device)

    val_loss = 0
    best_val_loss = 1e4
    for epoch in range(args.num_epoch):
        # train
        model.train()
        bar = tqdm(train_dl)
        for bi, (xb, yb) in enumerate(bar):
            model.zero_grad()

            loss = model.loss(xb.to(device), yb.to(device))
            loss.backward()
            optimizer.step()
            bar.set_description("{:2d}/{} loss: {:5.2f}, val_loss: {:5.2f}".format(
                epoch + 1, args.num_epoch, loss, val_loss))
            losses.append([epoch, bi, loss.item(), np.nan])

        # evaluation
        val_loss = _eval_model(model, device, dataloader=valid_dl, desc="eval").item()
        logger.info('epoch:{}, val_loss:{}'.format(epoch + 1, val_loss))
        # save losses
        losses[-1][-1] = val_loss
        _save_loss(losses, loss_path)

        # save model
        if not args.save_best_val_model or val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info("new best model(epoch: {}, val_loss: {})".format(epoch + 1, best_val_loss))
            _save_model(args.output_dir, model)

    # test
    test_loss = _eval_model(model, device, dataloader=test_dl, desc="test").item()
    last_loss = losses[-1][:]
    last_loss[-1] = test_loss
    losses.append(last_loss)
    _save_loss(losses, loss_path)
    logger.info("training completed. test loss: {:.2f}".format(test_loss))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the training files for the NER task.", )
    parser.add_argument('--output_dir', type=str, default="output_dir", help="the output directory for model files")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the predict sentence.")
    parser.add_argument('--num_epoch', type=int, default=20, help="number of epoch to train")
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='the L2 normalization parameter')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--device', type=str, default=None,
                        help='the training device: "cuda:0", "cpu:0". It will be auto-detected by default')
    parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length within training')
    parser.add_argument('--val_split', type=float, default=0.1, help='the split for the validation dataset')
    parser.add_argument('--test_split', type=float, default=0.1, help='the split for the testing dataset')
    parser.add_argument('--recovery', action="store_true", help="continue to train from the saved model in output dir")
    parser.add_argument('--save_best_val_model', action="store_true",
                        help="save the model whose validation score is smallest")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument('--embedding_dim', type=int, default=100, help='the dimension of the embedding layer')
    parser.add_argument('--hidden_dim', type=int, default=128, help='the dimension of the RNN hidden state')
    parser.add_argument('--num_rnn_layers', type=int, default=1, help='the number of RNN layers')
    parser.add_argument('--rnn_type', type=str, default="lstm", help='RNN type, choice: "lstm", "gru"')
    parser.add_argument('--predict_sentence', type=str, default="我哥常建生于1988年，是北京高级工程师。",
                        help="the sentence to be predicted")
    args = parser.parse_args()

    # Prepare NER task
    args.task_name = args.task_name.lower()
    args.output_dir = os.path.join(args.output_dir, args.rnn_type)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    init_logger(log_file=args.output_dir + '/{}-{}.log'.format('birnncrf', args.task_name))
    # Set seed
    seed_everything(args.seed)

    if args.do_train:
        train(args)
    if args.do_predict:
        predict(args)


if __name__ == "__main__":
    main()
