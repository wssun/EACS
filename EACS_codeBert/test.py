import torch
import torch.nn as nn
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
import bleu
from model import Seq2Seq
import sys

from transformers import AdamW, get_linear_schedule_with_warmup, \
    RobertaConfig, RobertaModel, RobertaTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class Example(object):
    def __init__(self, idx, ex_source, ab_source, target):
        self.idx = idx
        self.ex_source = ex_source
        self.ab_source = ab_source
        self.target = target


class InputFeatures(object):
    def __init__(self, example_id, ex_source_ids, ex_source_mask,
                 ab_source_ids, ab_source_mask, target_ids, target_mask):
        self.example_id = example_id

        self.ex_source_ids = ex_source_ids
        self.ex_source_mask = ex_source_mask

        self.ab_source_ids = ab_source_ids
        self.ab_source_mask = ab_source_mask

        self.target_ids = target_ids
        self.target_mask = target_mask


def read_examples(filename, ex_dataset, ab_dataset):
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f)):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx

            if np.sum(js['cleaned_seqs_pred']) == 0:
                ex_code = ' '.join(js[ex_dataset]).replace('\n', '')
            else:
                ex_code_list = []
                for index, i in enumerate(js['cleaned_seqs_pred']):
                    if i == 1:
                        ex_code_list.append(js[ex_dataset][index])
                ex_code = ' '.join(ex_code_list).replace('\n', '')

            ex_code = ' '.join(ex_code.strip().split())

            ab_code = ' '.join(js[ab_dataset]).replace('\n', '')
            ab_code = ' '.join(ab_code.strip().split())
            nl = ' '.join(js['cleaned_nl']).replace('\n', '')
            nl = ' '.join(nl.strip().split())

            examples.append(
                Example(
                    idx=idx,
                    ex_source=ex_code,
                    ab_source=ab_code,
                    target=nl
                )
            )

    return examples


def convert_examples_to_features(args, examples, tokenizer, code_length_ex=None, code_length_ab=None, stage=None):
    features = []
    for example_index, example in tqdm(enumerate(examples)):

        ex_source_tokens = tokenizer.tokenize(example.ex_source)[:code_length_ex - 2]
        ex_source_tokens = [tokenizer.cls_token] + ex_source_tokens + [tokenizer.sep_token]
        ex_source_ids = tokenizer.convert_tokens_to_ids(ex_source_tokens)
        ex_source_mask = [1] * (len(ex_source_tokens))
        padding_length = code_length_ex - len(ex_source_ids)
        ex_source_ids += [tokenizer.pad_token_id] * padding_length
        ex_source_mask += [0] * padding_length

        ab_source_tokens = tokenizer.tokenize(example.ab_source)[:code_length_ab - 2]
        ab_source_tokens = [tokenizer.cls_token] + ab_source_tokens + [tokenizer.sep_token]
        ab_source_ids = tokenizer.convert_tokens_to_ids(ab_source_tokens)
        ab_source_mask = [1] * (len(ab_source_tokens))
        padding_length = code_length_ab - len(ab_source_ids)
        ab_source_ids += [tokenizer.pad_token_id] * padding_length
        ab_source_mask += [0] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        features.append(
            InputFeatures(
                example_index,
                ex_source_ids,
                ex_source_mask,
                ab_source_ids,
                ab_source_mask,
                target_ids,
                target_mask
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
    parser.add_argument("--load_model_path", default=f'model/{language}/checkpoint-best-loss/pytorch_model.bin', type=str,
                        help="Path to trained model: Should contain the .bin files")

    parser.add_argument("--train_filename", default=f'../dataset/src-codeBERT-preds/{language}/train.jsonl', type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=f'../dataset/src-codeBERT-preds/{language}/valid.jsonl', type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=f'../dataset/src-codeBERT-preds/{language}/test.jsonl', type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length_ex", default=128, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_source_length_ab", default=256, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=64, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", default=False,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=False,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=True,
                        help="Whether to run test on the test set.")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--eval_steps", default=1000, type=int,
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
        # args.n_gpu = 1
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

    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)
    encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config)

    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

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
        train_examples = read_examples(args.train_filename, 'cleaned_seqs', 'cleaned_codes')
        train_features = convert_examples_to_features(args, train_examples, tokenizer,
                                                      code_length_ex=args.max_source_length_ex,
                                                      code_length_ab=args.max_source_length_ab,
                                                      stage='train')

        all_ex_source_ids = torch.tensor([f.ex_source_ids for f in train_features], dtype=torch.long)
        all_ex_source_mask = torch.tensor([f.ex_source_mask for f in train_features], dtype=torch.long)

        all_ab_source_ids = torch.tensor([f.ab_source_ids for f in train_features], dtype=torch.long)
        all_ab_source_mask = torch.tensor([f.ab_source_mask for f in train_features], dtype=torch.long)

        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_ex_source_ids, all_ex_source_mask,
                                   all_ab_source_ids, all_ab_source_mask,
                                   all_target_ids, all_target_mask)

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
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_loss, tr_ex_loss, tr_ab_loss, tr_acc, best_bleu = 0, 0, 0, 0, 0
        global_step, best_acc, best_loss = 0, 0, 1e6

        bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
        train_dataloader = cycle(train_dataloader)
        eval_flag = True
        print('start training...')

        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            ex_source_ids, ex_source_mask, ab_source_ids, ab_source_mask, target_ids, target_mask = batch

            loss, _, _ = model(
                ex_source_ids=ex_source_ids, ex_source_mask=ex_source_mask,
                ab_source_ids=ab_source_ids, ab_source_mask=ab_source_mask,
                target_ids=target_ids, target_mask=target_mask
            )

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            tr_loss += loss.item()

            train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)

            bar.set_description("loss {}".format(train_loss))

            nb_tr_examples += ex_source_ids.size(0)
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
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                eval_flag = False
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename, 'cleaned_seqs', 'cleaned_codes')
                    eval_features = convert_examples_to_features(args, eval_examples, tokenizer,
                                                                 code_length_ex=args.max_source_length_ex,
                                                                 code_length_ab=args.max_source_length_ab,
                                                                 stage='dev')
                    all_ex_source_ids = torch.tensor([f.ex_source_ids for f in eval_features], dtype=torch.long)
                    all_ex_source_mask = torch.tensor([f.ex_source_mask for f in eval_features], dtype=torch.long)

                    all_ab_source_ids = torch.tensor([f.ab_source_ids for f in eval_features], dtype=torch.long)
                    all_ab_source_mask = torch.tensor([f.ab_source_mask for f in eval_features], dtype=torch.long)

                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                    all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)

                    eval_data = TensorDataset(all_ex_source_ids, all_ex_source_mask, all_ab_source_ids,
                                              all_ab_source_mask, all_target_ids, all_target_mask)

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
                    ex_source_ids, ex_source_mask, ab_source_ids, ab_source_mask, target_ids, target_mask = batch

                    with torch.no_grad():
                        loss, _, _ = model(
                            ex_source_ids=ex_source_ids, ex_source_mask=ex_source_mask,
                            ab_source_ids=ab_source_ids, ab_source_mask=ab_source_mask,
                            target_ids=target_ids, target_mask=target_mask
                        )

                    eval_loss += loss.item()
                    eval_step += 1

                # Pring loss of dev dataset
                model.train()
                eval_loss = eval_loss / eval_step

                result = {'eval loss': round(eval_loss, 5),
                          'global step': global_step + 1,
                          'train loss': round(train_loss, 5)}

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
                    logger.info("  Best loss:%s", round(eval_loss, 5))
                    logger.info("  " + "*" * 20)
                    best_loss = eval_loss
                    # Save best checkpoint for best loss
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-loss')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

                # Calculate bleu
                if 'dev_bleu' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename, 'cleaned_seqs', 'cleaned_codes')
                    eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
                    eval_features = convert_examples_to_features(args, eval_examples, tokenizer,
                                                                 code_length_ex=args.max_source_length_ex,
                                                                 code_length_ab=args.max_source_length_ab,
                                                                 stage='test')

                    all_ex_source_ids = torch.tensor([f.ex_source_ids for f in eval_features], dtype=torch.long)
                    all_ex_source_mask = torch.tensor([f.ex_source_mask for f in eval_features], dtype=torch.long)

                    all_ab_source_ids = torch.tensor([f.ab_source_ids for f in eval_features], dtype=torch.long)
                    all_ab_source_mask = torch.tensor([f.ab_source_mask for f in eval_features], dtype=torch.long)

                    eval_data = TensorDataset(all_ex_source_ids, all_ex_source_mask,
                                              all_ab_source_ids, all_ab_source_mask)
                    dev_dataset['dev_bleu'] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                p = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    ex_source_ids, ex_source_mask, ab_source_ids, ab_source_mask = batch
                    with torch.no_grad():
                        preds = model(ex_source_ids=ex_source_ids, ex_source_mask=ex_source_mask,
                                      ab_source_ids=ab_source_ids, ab_source_mask=ab_source_mask)
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[:t.index(0)]
                            text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                            p.append(text)

                model.train()
                predictions = []
                with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(
                        os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
                    for ref, gold in zip(p, eval_examples):
                        predictions.append(str(gold.idx) + '\t' + ref)
                        f.write(str(gold.idx) + '\t' + ref + '\n')
                        f1.write(str(gold.idx) + '\t' + gold.target + '\n')

                (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold"))
                dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
                logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                logger.info("  " + "*" * 20)
                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s", dev_bleu)
                    logger.info("  " + "*" * 20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

    if args.do_test:
        files = []
        # if args.dev_filename is not None:
        #     files.append(args.dev_filename)
        if args.test_filename is not None:
            files.append(args.test_filename)

        for idx, file in enumerate(files):
            logger.info("Test file: {}".format(file))

            eval_examples = read_examples(file, 'cleaned_seqs', 'cleaned_codes')
            eval_features = convert_examples_to_features(args, eval_examples, tokenizer,
                                                         code_length_ex=args.max_source_length_ex,
                                                         code_length_ab=args.max_source_length_ab,
                                                         stage='test')

            all_ex_source_ids = torch.tensor([f.ex_source_ids for f in eval_features], dtype=torch.long)
            all_ex_source_mask = torch.tensor([f.ex_source_mask for f in eval_features], dtype=torch.long)

            all_ab_source_ids = torch.tensor([f.ab_source_ids for f in eval_features], dtype=torch.long)
            all_ab_source_mask = torch.tensor([f.ab_source_mask for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_ex_source_ids, all_ex_source_mask, all_ab_source_ids, all_ab_source_mask)

            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            p = []
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                ex_source_ids, ex_source_mask, ab_source_ids, ab_source_mask = batch
                with torch.no_grad():
                    preds = model(ex_source_ids=ex_source_ids, ex_source_mask=ex_source_mask,
                                  ab_source_ids=ab_source_ids, ab_source_mask=ab_source_mask)
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                        p.append(text)

            model.train()
            predictions = []

            with open(os.path.join(args.output_dir, "test_{}.output".format(str(idx))), 'w') as f, open(
                    os.path.join(args.output_dir, "test_{}.gold".format(str(idx))), 'w') as f1:
                for ref, gold in zip(p, eval_examples):
                    predictions.append(str(gold.idx) + '\t' + ref)
                    f.write(str(gold.idx) + '\t' + ref + '\n')
                    f1.write(str(gold.idx) + '\t' + gold.target + '\n')

            (goldMap, predictionMap) = bleu.computeMaps(predictions,
                                                        os.path.join(args.output_dir, "test_{}.gold".format(idx)))
            dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
            logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
            logger.info("  " + "*" * 20)


if __name__ == "__main__":
    language = sys.argv[1]
    main(language)
