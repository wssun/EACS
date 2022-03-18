import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np

import os
import argparse
import logging
import random
from tqdm import tqdm
from itertools import cycle
import json
import jsonlines
import platform

import sys

from model import SelectorNet

from transformers import AdamW, get_linear_schedule_with_warmup, \
    RobertaConfig, RobertaModel, RobertaTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

transformer_path = r'I:\project\codebert-base'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

sysstr = platform.system()


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class Example(object):
    def __init__(self, idx, source, labels):
        self.idx = idx
        self.source = source
        self.labels = labels


class InputFeatures(object):
    def __init__(self, example_id, source_ids, word_masks, stat_masks, labels):
        self.word_masks = word_masks
        self.example_id = example_id
        self.source_ids = source_ids
        self.stat_masks = stat_masks
        self.labels = labels


def read_examples(filename, code_type):
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f)):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx

            code = [i.replace('\n', ' ') for i in js[code_type]]
            labels = js['ex_labels']
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    labels=labels,
                )
            )

            # if idx > 100:
            #     break

    return examples


def convert_examples_to_features(examples, tokenizer, args, word_length=None, stat_length=None):
    features = []
    for example_index, example in tqdm(enumerate(examples)):
        source_ids = []
        word_masks = []

        # source
        for i in example.source[:stat_length]:
            stat_tokens = tokenizer.tokenize(i)[:word_length]
            stat_ids = tokenizer.convert_tokens_to_ids(stat_tokens)
            stat_mask = [1] * (len(stat_tokens))
            padding_length = word_length - len(stat_ids)
            stat_ids += [tokenizer.pad_token_id] * padding_length
            stat_mask += [0] * padding_length

            source_ids.append(stat_ids)
            word_masks.append(stat_mask)

        stat_masks = [1] * (len(source_ids))
        padding_length = stat_length - len(source_ids)
        stat_masks += [0] * padding_length
        source_ids += [[tokenizer.pad_token_id] * word_length] * padding_length
        word_masks += [[0] * word_length] * padding_length

        labels = example.labels[:stat_length]
        labels += [0] * padding_length

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                word_masks,
                stat_masks,
                labels
            )
        )
    return features


def main(language):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default='roberta', type=str,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=r'microsoft/codebert-base', type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=f'model/{language}', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")

    parser.add_argument("--train_filename", default=f'../dataset/src-codeBERT/{language}-ex/train.jsonl', type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=f'../dataset/src-codeBERT/{language}-ex/valid.jsonl', type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=f'../dataset/src-codeBERT/{language}-ex/test.jsonl', type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    parser.add_argument("--max_word_length", default=32, type=int)
    parser.add_argument("--max_stat_length", default=32, type=int)

    parser.add_argument("--word_hidden_size", default=128, type=int, help="")
    parser.add_argument("--stat_hidden_size", default=256, type=int, help="")
    parser.add_argument("--vocab_size", default=50265, type=int, help="")
    parser.add_argument("--embed_size", default=768, type=int, help="")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--do_train", default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=False,
                        help="Whether to run test on the test set.")
    parser.add_argument("--do_save_ex", default=False)

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--eval_steps", default=500, type=int,
                        help="")
    parser.add_argument("--train_steps", default=50000, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        # args.n_gpu = 2
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    if sysstr == 'Linux':
        config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        tokenizer = RobertaTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case)
        encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config)
    elif sysstr == 'Windows':
        config = RobertaConfig.from_pretrained(transformer_path)
        tokenizer = RobertaTokenizer.from_pretrained(transformer_path)
        encoder = RobertaModel.from_pretrained(transformer_path, config=config)

    word_embeddings_weight = encoder.embeddings.word_embeddings.weight
    model = SelectorNet(batch_size=args.train_batch_size, word_embeddings_weight=word_embeddings_weight,
                        word_hidden_size=args.word_hidden_size, stat_hidden_size=args.stat_hidden_size,
                        max_word_len=args.max_word_length, max_stat_len=args.max_stat_length,
                        vocab_size=args.vocab_size, embed_size=args.embed_size, num_classes=2,
                        imbalance_loss_fct=True)

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        train_examples = read_examples(args.train_filename, 'cleaned_seqs')
        train_features = convert_examples_to_features(train_examples, tokenizer, args,
                                                      word_length=args.max_word_length,
                                                      stat_length=args.max_stat_length)

        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_word_mask = torch.tensor([f.word_masks for f in train_features], dtype=torch.long)
        all_stat_mask = torch.tensor([f.stat_masks for f in train_features], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_source_ids, all_word_mask, all_stat_mask, all_labels)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps)

        num_train_optimization_steps = args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", num_train_optimization_steps * args.train_batch_size // len(train_examples))

        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, tr_acc, global_step, best_acc, best_loss = 0, 0, 0, 0, 0, 0, 1e6
        bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
        train_dataloader = cycle(train_dataloader)
        eval_flag = True
        print('start training...')

        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            source_ids, word_masks, stat_masks, labels = batch
            loss, _, num, active_labels, probs = model(source_ids, word_masks, stat_masks, labels)
            # print(loss)
            # print(_)
            # print(num)
            # print(active_labels)
            # print(probs)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss = np.round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)

            prediction = torch.argmax(probs, 1)
            cur_acc = (prediction[active_labels] == labels.view(-1)[active_labels]).sum().float()
            tr_acc += (cur_acc / num.sum()).cpu().detach().data.numpy()
            train_acc = np.round(tr_acc * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)

            bar.set_description("loss {}, acc {}".format(train_loss, train_acc))

            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                eval_flag = True

            if args.do_eval and ((global_step + 1) % args.eval_steps == 0) and eval_flag:
                # Eval model with dev dataset
                tr_loss, tr_acc = 0, 0
                nb_tr_examples, nb_tr_steps = 0, 0
                eval_flag = False
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename, 'cleaned_seqs')
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,
                                                                 word_length=args.max_word_length,
                                                                 stat_length=args.max_stat_length)
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_word_mask = torch.tensor([f.word_masks for f in eval_features], dtype=torch.long)
                    all_stat_mask = torch.tensor([f.stat_masks for f in eval_features], dtype=torch.long)
                    all_labels = torch.tensor([f.labels for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids, all_word_mask, all_stat_mask, all_labels)
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                # Start Evaling model
                model.eval()
                eval_loss = 0
                eval_acc, eval_step = 0, 0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, word_masks, stat_masks, labels = batch

                    with torch.no_grad():
                        loss, _, num, active_labels, probs = model(source_ids, word_masks, stat_masks, labels)

                    if args.n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    eval_loss += loss.item()

                    prediction = torch.argmax(probs, 1)
                    cur_acc = (prediction[active_labels] == labels.view(-1)[active_labels]).sum().float()
                    eval_acc += (cur_acc / num.sum()).cpu().detach().data.numpy()

                    eval_step += 1

                print('preds', prediction[active_labels])
                print('labels', labels.view(-1)[active_labels])

                # Pring loss of dev dataset
                model.train()
                eval_loss = eval_loss / eval_step
                eval_acc = eval_acc / eval_step
                result = {'eval_loss': np.round(eval_loss, 5),
                          'eval_acc': np.round(eval_acc, 5),
                          'global_step': global_step + 1,
                          'train_loss': np.round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                # save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)

                if eval_loss < best_loss:
                    logger.info("  Best loss:%s", np.round(eval_loss, 5))
                    logger.info("  " + "*" * 20)
                    best_loss = eval_loss
                    # Save best checkpoint for best loss
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-loss')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

                if eval_acc > best_acc:
                    logger.info("  Best acc:%s", np.round(eval_acc, 5))
                    logger.info("  " + "*" * 20)
                    best_acc = eval_acc

                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

    if args.do_test:
        files = []
        if args.dev_filename is not None:
            files.append(args.dev_filename)
        if args.test_filename is not None:
            files.append(args.test_filename)

        for idx, file in enumerate(files):
            logger.info("Test file: {}".format(file))

            eval_examples = read_examples(file, 'cleaned_seqs')
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args,
                                                         word_length=args.max_word_length,
                                                         stat_length=args.max_stat_length)
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_word_mask = torch.tensor([f.word_masks for f in eval_features], dtype=torch.long)
            all_stat_mask = torch.tensor([f.stat_masks for f in eval_features], dtype=torch.long)
            all_labels = torch.tensor([f.labels for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_source_ids, all_word_mask, all_stat_mask, all_labels)

            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()

            eval_acc,eval_step = 0, 0

            preds = []

            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids, word_masks, stat_masks, labels = batch

                with torch.no_grad():
                    _, _, num, active_labels, probs = model(source_ids, word_masks, stat_masks, labels)

                prediction = torch.argmax(probs, 1)

                preds.extend(prediction[active_labels].cpu().detach().data.numpy().tolist())

                cur_acc = (prediction[active_labels] == labels.view(-1)[active_labels]).sum().float()
                eval_acc += (cur_acc / num.sum()).cpu().detach().data.numpy()

                eval_step += 1

            if args.do_save_ex:
                save_ex_stat(file,preds, args.max_stat_length)

            print('preds', prediction[active_labels])
            print('labels', labels.view(-1)[active_labels])

            model.train()
            eval_acc = eval_acc / eval_step

            result = {'test acc': np.round(eval_acc, 5)}
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
            logger.info("  " + "*" * 20)

def save_ex_stat(filename, preds, max_stat_len):
    with open(filename, encoding="utf-8") as f,jsonlines.open('output-php.jsonl', mode='a') as writer:
        pred_idx = 0
        for idx, line in tqdm(enumerate(f)):
            line = line.strip()
            js = json.loads(line)

            stat_len = min(len(js['cleaned_seqs']),max_stat_len)

            js['cleaned_seqs_pred'] = preds[pred_idx,pred_idx+stat_len]

            writer.write(js)


if __name__ == "__main__":
    language = sys.argv[1]
    main(language)
