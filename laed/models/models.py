import torch
import torch.nn as nn
import torch.nn.functional as F
from laed.dataset.corpora import PAD, BOS, EOS
from torch.autograd import Variable
from laed import criterions
from laed.enc2dec.decoders import DecoderRNN
from laed.enc2dec.encoders import EncoderRNN
from laed.utils import INT, FLOAT, LONG, cast_type, get_bow
from laed import nn_lib
import numpy as np
from laed.models.model_bases import BaseModel
from laed.enc2dec.decoders import GEN
from laed.utils import Pack
import itertools


class DirVAE(BaseModel):
    def __init__(self, corpus, config):
        super(DirVAE, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.embed_size = config.embed_size
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.num_layer = config.num_layer
        self.dropout = config.dropout
        self.enc_cell_size = config.enc_cell_size
        self.dec_cell_size = config.dec_cell_size
        self.rnn_cell = config.rnn_cell
        self.max_dec_len = config.max_dec_len
        self.use_attn = config.use_attn
        self.beam_size = config.beam_size
        self.utt_type = config.utt_type
        self.bi_enc_cell = config.bi_enc_cell
        self.attn_type = config.attn_type
        self.enc_out_size = self.enc_cell_size * 2 if self.bi_enc_cell else self.enc_cell_size
        self.full_kl_step = 4200.0

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                      padding_idx=self.rev_vocab[PAD])

        self.x_encoder = EncoderRNN(self.embed_size, self.enc_cell_size,
                                    dropout_p=self.dropout,
                                    rnn_cell=self.rnn_cell,
                                    variable_lengths=self.config.fix_batch,
                                    bidirection=self.bi_enc_cell)

        # Dirichlet Topic Model prior
        self.h_dim = config.latent_size
        self.a = 1.*np.ones((1 , self.h_dim)).astype(np.float32)
        prior_mean = torch.from_numpy((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)
        prior_var = torch.from_numpy((((1.0 / self.a) * (1 - (2.0 / self.h_dim))).T +
                                 (1.0 / (self.h_dim * self.h_dim)) * np.sum(1.0 / self.a, 1)).T)
        prior_logvar = prior_var.log()

        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar)

        self.logvar_fc = nn.Sequential(
                        nn.Linear(self.enc_cell_size, np.maximum(config.latent_size * 2, 100)),
                        nn.Tanh(),
                        nn.Linear(np.maximum(config.latent_size * 2, 100), config.latent_size)
                        )
        self.mean_fc = nn.Sequential(
                        nn.Linear(self.enc_cell_size, np.maximum(config.latent_size * 2, 100)),
                        nn.Tanh(),
                        nn.Linear(np.maximum(config.latent_size * 2, 100), config.latent_size)
                        )
        self.mean_bn    = nn.BatchNorm1d(self.h_dim)                   # bn for mean
        self.logvar_bn  = nn.BatchNorm1d(self.h_dim)               # bn for logvar
        self.decoder_bn = nn.BatchNorm1d(self.vocab_size)

        self.logvar_bn.weight.requires_grad = False
        self.mean_bn.weight.requires_grad = False
        self.decoder_bn.weight.requires_grad = False

        self.logvar_bn.weight.fill_(1)
        self.mean_bn.weight.fill_(1)
        self.decoder_bn.weight.fill_(1)
        # self.q_y = nn.Linear(self.enc_out_size, self.h_dim)
        #self.cat_connector = nn_lib.GumbelConnector()

        # Prior for the Generation
        # self.z_mean  = nn.Sequential(
        #                 nn.Linear(self.enc_cell_size, np.maximum(config.latent_size * 2, 100)),
        #                 nn.Tanh(),
        #                 nn.Linear(np.maximum(config.latent_size * 2, 100), config.latent_size)
        #                 )

        # self.z_logvar = self.z_mean  = nn.Sequential(
        #                 nn.Linear(self.enc_cell_size, np.maximum(config.latent_size * 2, 100)),
        #                 nn.Tanh(),
        #                 nn.Linear(np.maximum(config.latent_size * 2, 100), config.latent_size)
        #                 )

        self.dec_init_connector = nn_lib.LinearConnector(config.latent_size, #+ self.h_dim,
                                                         self.dec_cell_size,
                                                         self.rnn_cell == 'lstm',
                                                         has_bias=False)

        self.decoder = DecoderRNN(self.vocab_size, self.max_dec_len,
                                  self.embed_size, self.dec_cell_size,
                                  self.go_id, self.eos_id,
                                  n_layers=1, rnn_cell=self.rnn_cell,
                                  input_dropout_p=self.dropout,
                                  dropout_p=self.dropout,
                                  use_attention=self.use_attn,
                                  attn_size=self.enc_cell_size,
                                  attn_mode=self.attn_type,
                                  use_gpu=self.use_gpu,
                                  embedding=self.embedding)

        self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)
        #self.cat_kl_loss = criterions.CatKLLoss()
        #self.cross_ent_loss = criterions.CrossEntropyoss()
        self.entropy_loss = criterions.Entropy()
        # self.log_py = nn.Parameter(torch.log(torch.ones(self.config.y_size,
        #                                                 self.config.k)/config.k),
        #                            requires_grad=True)
        # self.register_parameter('log_py', self.log_py)

        # self.log_uniform_y = Variable(torch.log(torch.ones(1) / config.k))
        # if self.use_gpu:
        #     self.log_uniform_y = self.log_uniform_y.cuda()

        # BOW loss
        self.bow_project = nn.Sequential(
            #nn.Linear(self.h_dim + config.latent_size, self.vocab_size),
            nn.Linear(config.latent_size, self.vocab_size),
            self.decoder_bn
        )

        self.kl_w = 0.0

    def print_top_words(self, n_top_words=10):
        emb = torch.nn.functional.softmax(self.bow_project[0].weight, 0).data.cpu().numpy().T
        print( '---------------Printing the Topics------------------')
        with open('topic-word.txt', 'w') as f:
            for i in range(len(emb)):
                print(" ".join([self.vocab[j+1]
                    for j in emb[i][1:].argsort()[:-n_top_words - 1:-1]]))
                f.write(" ".join([self.vocab[j+1]
                    for j in emb[i][1:].argsort()[:-n_top_words - 1:-1]]) + '\n')
        print( '---------------End of Topics------------------')

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = 0
        total_loss += loss.nll
        if self.config.use_reg_kl:
            total_loss += loss.reg_kl * self.kl_w
            #total_loss += loss.z_kld  
            #total_loss += loss.t_z_kld

        return total_loss
    
    def train_loss(self, loss, batch_cnt=None):
        total_loss = 0
        total_loss += loss.nll
        total_loss += loss.bow
        if self.config.use_reg_kl:
           total_loss += loss.reg_kl * self.kl_w
           #total_loss += loss.z_kld * self.kl_w
           #total_loss += loss.t_z_kld

        return total_loss

    def backward(self, batch_cnt, loss):
        total_loss = self.train_loss(loss, batch_cnt)
        total_loss.backward()

    def forward(self, data_feed, mode, global_t=1, gen_type='greedy', sample_n=1, return_latent=False):
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)

        # output encoder
        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        if type(x_last) is tuple:
            x_last = x_last[0].transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)
        x_topic, _ = get_bow(output_embedding)


        #topic posterior network
        posterior_mean   = self.mean_bn  (self.mean_fc  (x_last))          # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(x_last)) 
        # posterior_mean = self.mean_fc  (x_last)
        # posterior_logvar = self.logvar_fc(x_last)
        posterior_var    = posterior_logvar.exp()

        eps = posterior_mean.data.new().resize_as_(posterior_mean.data).normal_(0,1) # noise
        z = posterior_mean + posterior_var.sqrt() * eps                 # reparameterization
        self.p = F.softmax(z, -1)  

        # semantic posterior network
        # rec_mean = self.z_mean(x_last)
        # rec_logvar = self.z_logvar(x_last)
        # rec_var = rec_logvar.exp()

        # eps = rec_var.new_empty(rec_var.size()).normal_()
        # z = rec_mean + rec_var.sqrt() * eps

        # map sample to initial state of decoder
        #dec_init_state = self.dec_init_connector(torch.cat([z, self.p], -1))
        dec_init_state = self.dec_init_connector(z)
        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        self.bow_logits = self.bow_project(self.p)#torch.cat([z, self.p], -1))
        #self.z_bow_proj = self.bo

        if self.training:
            self.kl_w = min(global_t / self.full_kl_step, 1.0)
        else:
            self.kl_w = 1.0

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   dec_inputs, dec_init_state,
                                                   mode=mode, gen_type=gen_type,
                                                   beam_size=self.beam_size)
        # compute loss or return results
        if mode == GEN:
            return dec_ctx, labels
        else:
            # RNN reconstruction
            label_mask = torch.sign(labels).detach().float()
            bow_loss = -F.log_softmax(self.bow_logits.squeeze(), dim=1).gather(1, labels) * label_mask
            bow_loss = torch.sum(bow_loss, 1)
            self.avg_bow_loss  = torch.mean(bow_loss)

            nll = self.nll_loss(dec_outs, labels)
            prior_mean   = self.prior_mean.expand_as(posterior_mean)
            prior_var    = self.prior_var.expand_as(posterior_mean)
            prior_logvar = self.prior_logvar.expand_as(posterior_mean)
            var_division    = posterior_var  / prior_var
            diff            = posterior_mean - prior_mean
            diff_term       = diff * diff / prior_var
            logvar_division = prior_logvar - posterior_logvar
            # put KLD together
            KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.h_dim )
            #z_kld = self.gaussian_kld_normal(rec_mean, rec_logvar)
            self.avg_kld = torch.mean(KLD)
            #self.avg_z_kld = torch.mean(z_kld)
            #self.avg_z_topic_kld = torch.mean(topic_z_kld)
            #log_qy = F.log_softmax(z, -1)
            #avg_log_qy = torch.mean(log_qy, dim=0, keepdim=True)
            #mi = self.entropy_loss(avg_log_qy, unit_average=True)\
            #     - self.entropy_loss(log_qy, unit_average=True)

            results = Pack(nll=nll, reg_kl=self.avg_kld, bow=self.avg_bow_loss, kl_w=self.kl_w)#z_kld=self.avg_z_kld, t_z_kld=self.avg_z_topic_kld)

            #if return_latent:
            #    results['log_qy'] = log_qy
            #    results['dec_init_state'] = dec_init_state
            #    results['y_ids'] = self.p
            if return_latent:
                results['dec_init_state'] = dec_init_state
                results['log_qy'] = z
                retults['y_ids'] = self.p

            return results

    def sweep(self, data_feed, gen_type='greedy'):
        ctx_lens = data_feed['output_lens']
        batch_size = len(ctx_lens)
        out_utts = self.np2var(data_feed['outputs'], LONG)

        # output encoder
        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        x_last = x_last.transpose(0, 1).contiguous().view(-1, self.enc_out_size)

        # posterior network
        posterior_mean   = self.mean_bn  (self.mean_fc  (x_last))          # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(x_last)) 
        posterior_var    = posterior_logvar.exp()
        # switch that controls the sampling
        # sample_y, y_id = self.cat_connector(qy_logits, 1.0, self.use_gpu,
        #                                     hard=True, return_max_id=True)
        # y_id = y_id.view(-1, self.config.y_size)
        # start_y_id = y_id[0]
        # end_y_id = y_id[batch_size-1]

        # start sweeping
        # all_y_ids = [start_y_id]
        # for idx in range(self.config.y_size):
        #     mask = torch.zeros(self.config.y_size)
        #     mask[0:idx+1] = 1.0
        #     neg_mask = 1 - mask
        #     mask = cast_type(Variable(mask), LONG, self.use_gpu)
        #     neg_mask = cast_type(Variable(neg_mask), LONG, self.use_gpu)
        #     temp_y = neg_mask * start_y_id + mask * end_y_id
        #     all_y_ids.append(temp_y)
        # num_steps = len(all_y_ids)
        # all_y_ids = torch.cat(all_y_ids, dim=0).view(num_steps, -1)

        # sample_y = cast_type(Variable(torch.zeros((num_steps*self.config.y_size, self.config.k))), FLOAT, self.use_gpu)
        # sample_y.scatter_(1, all_y_ids.view(-1, 1), 1.0)
        # sample_y = sample_y.view(-1, self.config.k * self.config.y_size)
        # batch_size = num_steps
        eps = posterior_mean.data.new().resize_as_(posterior_mean.data).normal_(0,1) # noise
        z = posterior_mean + posterior_var.sqrt() * eps                 # reparameterization
        start_z_code = z[0]
        end_z_code = z[batch_size - 1]
        all_z_codes = [start_z_code]
        delta = 0.2
        for idx in range(5):
            weight = (idx + 1 ) * delta
            all_z_codes.append(start_z_code * (1.0 - weight) + weight * end_z_code )
        num_steps = len(all_z_codes)
        all_z_codes = torch.cat(all_z_codes, dim=0).view(num_steps, -1)
        batch_size = num_steps
        #start sweeping

        # map sample to initial state of decoder
        dec_init_state = self.dec_init_connector(all_z_codes)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   dec_inputs, dec_init_state,
                                                   mode=GEN, gen_type=gen_type,
                                                   beam_size=self.beam_size)
        # compute loss or return results
        return dec_ctx, labels, all_z_codes

    def enumerate(self, repeat=1, gen_type='greedy'):

        # do something here. For each y, we enumerate from 0 to K
        # and take the expectation of other values.
        batch_size = self.config.y_size * self.config.k * repeat
        sample_y = cast_type(Variable(torch.zeros((batch_size,
                                                   self.config.y_size,
                                                   self.config.k))),
                             FLOAT, self.use_gpu)
        sample_y += 1.0/self.config.k

        for y_id in range(self.config.y_size):
            for k_id in range(self.config.k):
                for r_id in range(repeat):
                    idx = y_id*self.config.k + k_id*repeat + r_id
                    sample_y[idx, y_id] = 0.0
                    sample_y[idx, y_id, k_id] = 1.0

        # map sample to initial state of decoder
        sample_y = sample_y.view(-1, self.config.k * self.config.y_size)
        dec_init_state = self.dec_init_connector(sample_y)

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   None, dec_init_state,
                                                   mode=GEN, gen_type=gen_type,
                                                   beam_size=self.beam_size)
        # compute loss or return results
        return dec_ctx

    def exp_enumerate(self, repeat=1, gen_type='greedy'):

        # do something here. For each y, we enumerate from 0 to K
        # and take the expectation of other values.
        batch_size = np.power(self.config.k, self.config.y_size) * repeat
        sample_y = cast_type(Variable(torch.zeros((batch_size*self.config.y_size,
                                                   self.config.k))),
                             FLOAT, self.use_gpu)
        d = dict((str(i), range(self.config.k)) for i in range(self.config.y_size))
        all_y_ids = []
        for combo in itertools.product(*[d[k] for k in sorted(d.keys())]):
            all_y_ids.append(list(combo))
        np_y_ids = np.array(all_y_ids)
        np_y_ids = self.np2var(np_y_ids, LONG)
        # map sample to initial state of decoder
        sample_y.scatter_(1, np_y_ids.view(-1, 1), 1.0)
        sample_y = sample_y.view(-1, self.config.k * self.config.y_size)
        dec_init_state = self.dec_init_connector(sample_y)

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   None, dec_init_state,
                                                   mode=GEN, gen_type=gen_type,
                                                   beam_size=self.beam_size)
        return dec_ctx, all_y_ids

    def gaussian_kld_normal(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar), 1)
        return kld

    def gaussian_kld(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                                - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                                - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
        return kld
