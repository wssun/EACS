# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy

from utils import matrix_mul, element_wise_mul


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class WordNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, word_embeddings_weight=None):
        super(WordNet, self).__init__()

        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        if word_embeddings_weight != None:
            self.word_embeddings.from_pretrained(word_embeddings_weight)

        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)

        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.word_bias.data.zero_()
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        output = self.word_embeddings(input)
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output)
        output = element_wise_mul(f_output, output.permute(1, 0))

        return output, h_output


class StatNet(nn.Module):
    def __init__(self, stat_hidden_size, word_hidden_size, num_classes):
        super(StatNet, self).__init__()

        self.stat_weight = nn.Parameter(torch.Tensor(2 * stat_hidden_size, 2 * stat_hidden_size))
        self.stat_bias = nn.Parameter(torch.Tensor(1, 2 * stat_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * stat_hidden_size, 1))

        self.gru = nn.GRU(2 * word_hidden_size, stat_hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * stat_hidden_size, num_classes)

        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.stat_weight.data.normal_(mean, std)
        self.stat_bias.data.zero_()
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        f_output, h_output = self.gru(input, hidden_state)
        f_output = f_output.permute(1, 0, 2)

        logits = []
        probs = []

        for f in f_output:
            logit = self.fc(f)
            logits.append(logit)

            p = F.sigmoid(logit)
            probs.append(p)

        logits = torch.cat(logits, dim=0)
        probs = torch.cat(probs, dim=0)

        return logits, probs, h_output


class SelectorNet(nn.Module):
    def __init__(self, batch_size, word_embeddings_weight, word_hidden_size, stat_hidden_size, max_word_len,
                 max_stat_len, vocab_size, embed_size, num_classes, imbalance_loss_fct):
        super(SelectorNet, self).__init__()

        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.stat_hidden_size = stat_hidden_size

        self.max_word_len = max_word_len
        self.max_stat_len = max_stat_len

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_classes = num_classes

        self.word_embeddings_weight = word_embeddings_weight

        self.word_net = WordNet(vocab_size, embed_size, word_hidden_size, word_embeddings_weight)
        self.stat_net = StatNet(stat_hidden_size, word_hidden_size, num_classes)

        if imbalance_loss_fct:
            self.loss_fct = FocalLoss(2)
        else:
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.stat_hidden_state = torch.zeros(2, batch_size, self.stat_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.stat_hidden_state = self.stat_hidden_state.cuda()

    def forward(self, input, w_mask, s_mask, labels):
        batch_size = input.size(0)
        self._init_hidden_state(batch_size)

        output_list = []
        input = input.permute(1, 0, 2)

        for i in input:
            output, self.word_hidden_state = self.word_net(i.permute(1, 0), self.word_hidden_state.data)
            output_list.append(output)

        outputs = torch.cat(output_list, 0)
        logits, probs, self.stat_hidden_state = self.stat_net(outputs, self.stat_hidden_state.data)

        active_mask = s_mask.ne(0).view(-1) == 1
        # loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        # loss = loss_fct(logits.view(-1, logits.size(-1))[active_loss],
        #                 labels.view(-1)[active_loss])

        loss = self.loss_fct(logits.view(-1, logits.size(-1))[active_mask],
                             labels.view(-1)[active_mask])

        net_outputs = loss, loss * active_mask.sum(), active_mask.sum(), active_mask, probs

        return net_outputs


class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """

    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None,
                 eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head, self.encoder.embeddings.word_embeddings)

    def forward(self, ex_source_ids=None, ex_source_mask=None, ab_source_ids=None,
                ab_source_mask=None, target_ids=None, target_mask=None, args=None):

        outputs_ex = self.encoder(ex_source_ids, attention_mask=ex_source_mask)
        outputs_ab = self.encoder(ab_source_ids, attention_mask=ab_source_mask)

        outputs = torch.cat((outputs_ex[0], outputs_ab[0]), 1)

        source_mask = torch.cat((ex_source_mask, ab_source_mask), 1)

        encoder_output = outputs.permute([1, 0, 2]).contiguous()
        if target_ids is not None:
            attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1, 0, 2]).contiguous()
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                               memory_key_padding_mask=(1 - source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss, loss * active_loss.sum(), active_loss.sum()
            return outputs
        else:
            # Predict
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(ex_source_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                       memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence


class JoinNet(nn.Module):
    def __init__(self, batch_size, word_embeddings_weight, word_hidden_size, stat_hidden_size,
                 max_word_len, max_stat_len, vocab_size, embed_size, num_classes, imbalance_loss_fct,
                 encoder, decoder, config, beam_size, max_length, sos_id, eos_id, padding_id, device):
        super(JoinNet, self).__init__()

        self.batch_size = batch_size
        self.word_embeddings_weight = word_embeddings_weight
        self.word_hidden_size = word_hidden_size
        self.stat_hidden_size = stat_hidden_size
        self.max_word_len = max_word_len
        self.max_stat_len = max_stat_len
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.imbalance_loss_fct = imbalance_loss_fct

        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

        self.padding_id = padding_id

        self.device = device

        self.selector_net = SelectorNet(batch_size, word_embeddings_weight,
                                        word_hidden_size, stat_hidden_size,
                                        max_word_len, max_stat_len,
                                        vocab_size, embed_size, num_classes,
                                        imbalance_loss_fct)
        self.seq2seq = Seq2Seq(encoder, decoder, config, beam_size=beam_size,
                               max_length=max_length, sos_id=sos_id, eos_id=eos_id)

    def get_ex_input(self, ex_source_ids, select_probs, w_mask, s_mask, ex_length):
        # select_probs是否需要在no grad下进行
        sel_probs = select_probs.cpu().detach().data
        sel_probs = torch.argmax(sel_probs, 1)
        activate_select = s_mask.ne(0).view(-1) == 1
        activate_probs = sel_probs[activate_select]

        ex_inputs = []
        ex_mask = []

        ex_code_len = ex_length * ex_source_ids.size(-1)

        for i in range(s_mask.size(0)):
            input = []
            # mask = []

            end_idx = s_mask[i].cpu().sum().int().numpy().tolist()

            for j in range(0, min(end_idx, ex_length)):
                if activate_probs[j]:
                    input_activate = w_mask[i, j, :].cpu().ne(0).view(-1) == 1
                    input.extend(ex_source_ids[i, j, :][input_activate].cpu().numpy().tolist())
                    # mask.extend(w_mask[i, j, :][input_activate].cpu().numpy().tolist())

            if len(input) == 0:
                for k in range(ex_length):
                    input.extend(ex_source_ids[i, k, :].cpu().numpy().tolist())
                    # mask.extend(w_mask[i, k, :].cpu().numpy().tolist())

            input = input[:ex_code_len - 2]
            input = [self.sos_id] + input + [self.eos_id]
            padding_len = ex_code_len - len(input)
            mask = [1] * len(input)
            mask += [0] * padding_len

            input += [self.padding_id] * padding_len

            ex_inputs.append(input)
            ex_mask.append(mask)

        ex_inputs = torch.tensor(ex_inputs, dtype=torch.long)
        ex_mask = torch.tensor(ex_mask, dtype=torch.long)

        return ex_inputs, ex_mask

    def forward(self, ex_source_ids, w_mask, s_mask, labels,
                ab_source_ids, ab_source_mask, target_ids, target_mask,
                ex_length, stage, args):
        ex_loss, ex_loss_sum, ex_activate_num, ex_active_mask, select_probs = self.selector_net(ex_source_ids,
                                                                                                w_mask, s_mask,
                                                                                                labels)
        ex_inputs, ex_mask = self.get_ex_input(ex_source_ids, select_probs, w_mask, s_mask, ex_length)

        ex_inputs = ex_inputs.to(self.device)
        ex_mask = ex_mask.to(self.device)

        if stage == 'test':
            preds = self.seq2seq(ex_inputs, ex_mask, ab_source_ids, ab_source_mask, None, None, args)
            outputs = ex_activate_num, ex_active_mask, select_probs, preds
        else:
            ab_loss, ab_loss_sum, ab_activate_num, = self.seq2seq(ex_inputs, ex_mask, ab_source_ids,
                                                                  ab_source_mask, target_ids, target_mask,
                                                                  args)
            loss = ex_loss + ab_loss
            outputs = loss, ex_loss, ex_activate_num, ex_active_mask, select_probs, ab_loss, ab_loss_sum, ab_activate_num

        return outputs
