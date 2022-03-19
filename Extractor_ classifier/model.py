import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import matrix_mul, element_wise_mul


class FocalLoss(nn.Module):
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
        loss = self.loss_fct(logits.view(-1, logits.size(-1))[active_mask],
                             labels.view(-1)[active_mask])

        net_outputs = loss, loss * active_mask.sum(), active_mask.sum(), active_mask, probs

        return net_outputs
