import torch
import torch.nn as nn

from onmt.decoders.decoder import InputFeedRNNDecoder
from onmt.utils.misc import sequence_mask
from onmt.utils.misc import aeq
from onmt.models.stacked_rnn import StackedLSTM, StackedGRU
from onmt.utils.rnn_factory import rnn_factory
from onmt.modules import context_gate_factory, GlobalAttention
from onmt.decoders.decoder import DecoderBase

try:
    from src.i3dpt import get_padding_shape
except ImportError:
    get_padding_shape = None


class AFRAttentionWeighting(nn.Module):
    def __init__(self, in_feats, intr_feats, hid):
        super(AFRAttentionWeighting, self).__init__()
        self.feat2intr = nn.Linear(in_feats, intr_feats)
        self.hid2intr = nn.Linear(hid, intr_feats, bias=False)
        self.intr2one = nn.Linear(intr_feats, 1)
        self.sm = nn.Softmax(dim=0)
        self.tanh = nn.Tanh()

    def forward(self, inp, hid, mask=None):
        unscaled = self.intr2one(self.tanh(
            self.feat2intr(inp) + self.hid2intr(hid)))
        if mask is not None:
            unscaled.masked_fill_(~mask, -float('inf'))
        return self.sm(unscaled)


class AFRAttention(nn.Module):
    def __init__(self, in_feats, intr_feats, hid):
        super(AFRAttention, self).__init__()
        self.weighting = AFRAttentionWeighting(in_feats, intr_feats, hid)

    def forward(self, inp, keys, hid, mask=None):
        weighting = self.weighting(keys, hid, mask)
        ctx = torch.bmm(inp.permute(1, 2, 0), weighting.permute(1, 0, 2))\
            .permute(2, 0, 1)
        return ctx, weighting


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a=-0.01, b=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.uniform_(param, a=-0.01, b=0.01)
            elif "bias" in name:
                nn.init.constant_(param, 0)
            else:
                assert False


class DualAFRDecoder(InputFeedRNNDecoder):
    def __init__(self, *args, **kwargs):
        super(DualAFRDecoder, self).__init__(*args, **kwargs)
        del self.attn
        self.rnn = nn.LSTM(self.embeddings.embedding_size + 512, self.hidden_size)
        # self.c0 = nn.Linear(512, self.embeddings.embedding_size, bias=False)
        self.c3d2c = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 512, bias=False))
        self.fv2h = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, self.hidden_size, bias=False),
            nn.Tanh(),
            nn.Dropout())
        self.fv2c = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, self.hidden_size, bias=False),
            nn.Tanh(),
            nn.Dropout())
        self.v = nn.Linear(512, 300)

        self.s = AFRAttention(512, 512, 512)
        self.s2o = nn.Linear(512, 300)
        self.e2o = nn.Linear(300, 300)
        self.tanh = nn.Tanh()
        # self.z = AFRAttention(512, 512, self.hidden_size * self.num_layers)

        self.apply(init_weights)

    def init_state(self, src, memory_bank, encoder_final):
        """Initialize decoder state with last state of the encoder."""
        encoder_final = self.c3d2c(encoder_final)
        h = self.fv2h(encoder_final)
        c = self.fv2c(encoder_final)
        self.h0 = encoder_final.clone()
        self.state["hidden"] = (h, c)

        # Init the input feed.
        self.state["coverage"] = None

        self.initial = True

    def map_state(self, fn):
        self.state["hidden"] = tuple(fn(h, 1) for h in self.state["hidden"])
        self.h0 = fn(self.h0, 1)

    def detach_state(self):
        self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
        self.h0 = self.h0.detach()

    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        _, tgt_batch, _ = tgt.size()
        # END Additional args check.

        dec_outs = []
        attns = {"std": []}
        if self.copy_attn is not None or self._reuse_copy_attn:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        # if self.initial:
        #     emb = self.dropout(self.embeddings(tgt[1:]))
        #     z0 = self.c0(self.h0)
        #     self.initial = False
        #     emb = torch.cat([z0, emb], dim=0)
        # else:
        #     emb = self.dropout(self.embeddings(tgt))
        emb = self.dropout(self.embeddings(tgt))

        assert emb.dim() == 3  # len x batch x embedding_dim

        dec_state = self.state["hidden"]
        coverage = self.state["coverage"].squeeze(0) \
            if self.state["coverage"] is not None else None

        bsz, nF, nz, nx, ny = memory_bank[-1].shape
        # mask = sequence_mask(memory_lengths[:, -1], max_len=nz)\
        #     .view(bsz, 1, nz, 1, 1)\
        #     .repeat(1, 1, 1, nx, ny)\
        #     .view(bsz, 1, -1)\
        #     .permute(2, 0, 1)
        # # keys_i = torch.cat(memory_bank, 1)
        # keys_i = memory_bank[2].flatten(2).permute(2, 0, 1)
        # # keys_i = keys_i.view(bsz, nF*len(memory_bank), -1).permute(2, 0, 1)

        for emb_t in emb.split(1):
            # if self.initial:
            #     self.initial = False
            #     dec_outs.append(self.dropout(dec_state[0].squeeze(0)))
            #     continue
            mb_3 = self.dropout(memory_bank[3].flatten(2).permute(2, 0, 1))
            s, p_attn = self.s(mb_3,  # a_seq_over_i,
                               mb_3,
                               dec_state[0].view(bsz, -1))
            # s = self.dropout(s)
            # s = self.dropout(self.h0ctx2ctx(torch.cat([s, self.h0], -1)))
            # s_seq_over_L = torch.cat(
            #     [s[:, :, i*nF:(i+1)*nF] for i in range(len(memory_bank))], 0)
            # keys_l = s_seq_over_L
            # z, l_attn = self.z(keys_l,
            #                    keys_l,
            #               dec_state[0].view(bsz, -1))
            # decoder_input = torch.cat([emb_t, z], 2)
            # p_attn = torch.zeros((1, 1, 1))
            rnn_output, dec_state = self.rnn(
                torch.cat([emb_t, self.dropout(s)], -1),
                dec_state)
            # p_attn = torch.zeros((1, 1, 1))
            decoder_output = self.dropout(self.tanh(
                self.v(self.dropout(rnn_output.squeeze(0))) \
                + self.e2o(emb_t.squeeze(0)) \
                + self.s2o(self.dropout(s.squeeze(0)))))

            dec_outs += [decoder_output]
            attns["std"] += [p_attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = p_attn if coverage is None else p_attn + coverage
                attns["coverage"] += [coverage]

            if self.copy_attn is not None:
                _, copy_attn = self.copy_attn(
                    decoder_output, memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._reuse_copy_attn:
                attns["copy"] = attns["std"]

        return dec_state, dec_outs, attns

    def forward(self, tgt, memory_bank, memory_lengths=None, step=None):
        """
        Args:
            tgt (LongTensor): sequences of padded tokens
                 ``(tgt_len, batch, nfeats)``.
            memory_bank (FloatTensor): vectors from the encoder
                 ``(src_len, batch, hidden)``.
            memory_lengths (LongTensor): the padded source lengths
                ``(batch,)``.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * dec_outs: output from the decoder (after attn)
              ``(tgt_len, batch, hidden)``.
            * attns: distribution over src at each tgt
              ``(tgt_len, batch, src_len)``.
        """

        # print(memory_bank)

        dec_state, dec_outs, attns = self._run_forward_pass(
            tgt, memory_bank, memory_lengths=memory_lengths)

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["coverage"] = None
        if "coverage" in attns:
            self.state["coverage"] = attns["coverage"][-1].unsqueeze(0)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            for k in attns:
                if type(attns[k]) == list:
                    if len(attns[k]) > 0:
                        attns[k] = torch.stack(attns[k])
        return dec_outs, attns

#
# class DualAFRDecoder(DecoderBase):
#     """Base recurrent attention-based decoder class.
#
#     Specifies the interface used by different decoder types
#     and required by :class:`~onmt.models.NMTModel`.
#
#
#     .. mermaid::
#
#        graph BT
#           A[Input]
#           subgraph RNN
#              C[Pos 1]
#              D[Pos 2]
#              E[Pos N]
#           end
#           G[Decoder State]
#           H[Decoder State]
#           I[Outputs]
#           F[memory_bank]
#           A--emb-->C
#           A--emb-->D
#           A--emb-->E
#           H-->C
#           C-- attn --- F
#           D-- attn --- F
#           E-- attn --- F
#           C-->I
#           D-->I
#           E-->I
#           E-->G
#           F---I
#
#     Args:
#        rnn_type (str):
#           style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
#        bidirectional_encoder (bool) : use with a bidirectional encoder
#        num_layers (int) : number of stacked layers
#        hidden_size (int) : hidden size of each layer
#        attn_type (str) : see :class:`~onmt.modules.GlobalAttention`
#        attn_func (str) : see :class:`~onmt.modules.GlobalAttention`
#        coverage_attn (str): see :class:`~onmt.modules.GlobalAttention`
#        context_gate (str): see :class:`~onmt.modules.ContextGate`
#        copy_attn (bool): setup a separate copy attention mechanism
#        dropout (float) : dropout value for :class:`torch.nn.Dropout`
#        embeddings (onmt.modules.Embeddings): embedding module to use
#        reuse_copy_attn (bool): reuse the attention for copying
#     """
#
#     def __init__(self, rnn_type, bidirectional_encoder, num_layers,
#                  hidden_size, attn_type="general", attn_func="softmax",
#                  coverage_attn=False, context_gate=None,
#                  copy_attn=False, dropout=0.0, embeddings=None,
#                  reuse_copy_attn=False):
#         super(DualAFRDecoder, self).__init__()
#
#         self.bidirectional_encoder = bidirectional_encoder
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.embeddings = embeddings
#         self.dropout = nn.Dropout(dropout)
#
#         # Decoder state
#         self.state = {}
#
#         # Build the RNN.
#         self.rnn = self._build_rnn(rnn_type,
#                                    input_size=self._input_size,
#                                    hidden_size=hidden_size,
#                                    num_layers=num_layers,
#                                    dropout=dropout)
#
#         # Set up the context gate.
#         self.context_gate = None
#         if context_gate is not None:
#             self.context_gate = context_gate_factory(
#                 context_gate, self._input_size,
#                 hidden_size, hidden_size, hidden_size
#             )
#
#         # # Set up the standard attention.
#         self._coverage = coverage_attn
#         # self.attn = GlobalAttention(
#         #     hidden_size, coverage=coverage_attn,
#         #     attn_type=attn_type, attn_func=attn_func
#         # )
#         self.attn = AFRAttention(512, 512, 512)
#
#         self.v = nn.Linear(512, 300)
#
#         if copy_attn and not reuse_copy_attn:
#             self.copy_attn = GlobalAttention(
#                 hidden_size, attn_type=attn_type, attn_func=attn_func
#             )
#         else:
#             self.copy_attn = None
#
#         self._reuse_copy_attn = reuse_copy_attn and copy_attn
#         self.fv2h = nn.Sequential(
#             nn.Linear(512, self.hidden_size, bias=False),
#             nn.Tanh())
#         self.fv2c = nn.Sequential(
#             nn.Linear(512, self.hidden_size, bias=False),
#             nn.Tanh())
#         # self.w = nn.Linear(1024, 512)
#         self.outp = nn.Linear(1024, 512)
#         self.c0 = nn.Linear(512, 512)
#
#         self.apply(init_weights)
#
#     @classmethod
#     def from_opt(cls, opt, embeddings):
#         """Alternate constructor."""
#         return cls(
#             opt.rnn_type,
#             opt.brnn,
#             opt.dec_layers,
#             opt.dec_rnn_size,
#             opt.global_attention,
#             opt.global_attention_function,
#             opt.coverage_attn,
#             opt.context_gate,
#             opt.copy_attn,
#             opt.dropout,
#             embeddings,
#             opt.reuse_copy_attn)
#
#     def init_state(self, src, memory_bank, encoder_final):
#         """Initialize decoder state with last state of the encoder."""
#         # def _fix_enc_hidden(hidden):
#         #     # The encoder hidden is  (layers*directions) x batch x dim.
#         #     # We need to convert it to layers x batch x (directions*dim).
#         #     if self.bidirectional_encoder:
#         #         hidden = torch.cat([hidden[0:hidden.size(0):2],
#         #                             hidden[1:hidden.size(0):2]], 2)
#         #     return hidden
#
#         # if isinstance(encoder_final, tuple):  # LSTM
#         #     self.state["hidden"] = tuple(_fix_enc_hidden(enc_hid)
#         #                                  for enc_hid in encoder_final)
#
#         encoder_final = self.dropout(encoder_final)
#
#         self.state["hidden"] = (
#             self.fv2h(encoder_final),
#             self.fv2c(encoder_final.clone()))
#         # else:  # GRU
#         #     self.state["hidden"] = (_fix_enc_hidden(encoder_final), )
#
#         # Init the input feed.
#         batch_size = self.state["hidden"][0].size(1)
#         h_size = (batch_size, self.hidden_size)
#         self.h0 = encoder_final
#         self.state["input_feed"] = self.dropout(self.c0(encoder_final))
#         # self.state["input_feed"] = self.dropout(self.outp(
#         #         torch.cat([self.state["hidden"][0].squeeze(0),
#         #                    torch.zeros_like(self.state["hidden"][0]).squeeze(0),
#         #                    self.h0.squeeze(0)], -1))).unsqueeze(0)
#         # self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
#         self.state["coverage"] = None
#
#     def map_state(self, fn):
#         self.state["hidden"] = tuple(fn(h, 1) for h in self.state["hidden"])
#         self.state["input_feed"] = fn(self.state["input_feed"], 1)
#         self.h0 = fn(self.h0, 1)
#
#     def detach_state(self):
#         self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
#         self.state["input_feed"] = self.state["input_feed"].detach()
#         self.h0.detach()
#
#     def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None):
#         """
#         See StdRNNDecoder._run_forward_pass() for description
#         of arguments and return values.
#         """
#         # Additional args check.
#         input_feed = self.state["input_feed"]
#         _, input_feed_batch, _ = input_feed.size()
#         _, tgt_batch, _ = tgt.size()
#         aeq(tgt_batch, input_feed_batch)
#         # END Additional args check.
#
#         dec_outs = []
#         attns = {"std": []}
#         if self.copy_attn is not None or self._reuse_copy_attn:
#             attns["copy"] = []
#         if self._coverage:
#             attns["coverage"] = []
#
#         emb = self.embeddings(tgt)
#         assert emb.dim() == 3  # len x batch x embedding_dim
#
#         dec_state = self.state["hidden"]
#         coverage = self.state["coverage"].squeeze(0) \
#             if self.state["coverage"] is not None else None
#
#         # Input feed concatenates hidden state with
#         # input at every time step.
#         for emb_t in emb.split(1):
#             # input_feed = self.w(torch.cat([input_feed, self.h0.squeeze(0)], -1))
#             decoder_input = torch.cat([emb_t.squeeze(0), self.h0.squeeze(0) + input_feed.squeeze(0)], 1)
#             rnn_output, dec_state = self.rnn(decoder_input, dec_state)
#
#             memory_bank_2 = memory_bank[2].flatten(2).permute(2, 0, 1)
#             attn_output, p_attn = self.attn(
#                 memory_bank_2, memory_bank_2, rnn_output)
#             decoder_output = self.dropout(self.outp(
#                 torch.cat([rnn_output, attn_output.squeeze(0)], -1)))
#
#             # decoder_output = decoder_output.squeeze(0)
#
#             # decoder_output, p_attn = self.attn(
#             #     rnn_output,
#             #     memory_bank.transpose(0, 1),
#             #     memory_lengths=memory_lengths)
#             if self.context_gate is not None:
#                 # TODO: context gate should be employed
#                 # instead of second RNN transform.
#                 decoder_output = self.context_gate(
#                     decoder_input, rnn_output, decoder_output
#                 )
#             # decoder_output = self.dropout(decoder_output)
#             input_feed = decoder_output.unsqueeze(0)
#             self.state["input_feed"] = input_feed
#
#             decoder_output = self.dropout(self.v(decoder_output))
#
#             dec_outs += [decoder_output]
#             attns["std"] += [p_attn]
#
#             # Update the coverage attention.
#             if self._coverage:
#                 coverage = p_attn if coverage is None else p_attn + coverage
#                 attns["coverage"] += [coverage]
#
#             if self.copy_attn is not None:
#                 _, copy_attn = self.copy_attn(
#                     decoder_output, memory_bank.transpose(0, 1))
#                 attns["copy"] += [copy_attn]
#             elif self._reuse_copy_attn:
#                 attns["copy"] = attns["std"]
#
#         return dec_state, dec_outs, attns
#
#     def _build_rnn(self, rnn_type, input_size,
#                    hidden_size, num_layers, dropout):
#         assert rnn_type != "SRU", "SRU doesn't support input feed! " \
#             "Please set -input_feed 0!"
#         stacked_cell = StackedLSTM if rnn_type == "LSTM" else StackedGRU
#         return stacked_cell(num_layers, input_size, hidden_size, dropout)
#
#     @property
#     def _input_size(self):
#         """Using input feed by concatenating input with attention vectors."""
#         return self.embeddings.embedding_size + self.hidden_size
#
#     def forward(self, tgt, memory_bank, memory_lengths=None, step=None):
#         """
#         Args:
#             tgt (LongTensor): sequences of padded tokens
#                  ``(tgt_len, batch, nfeats)``.
#             memory_bank (FloatTensor): vectors from the encoder
#                  ``(src_len, batch, hidden)``.
#             memory_lengths (LongTensor): the padded source lengths
#                 ``(batch,)``.
#
#         Returns:
#             (FloatTensor, dict[str, FloatTensor]):
#
#             * dec_outs: output from the decoder (after attn)
#               ``(tgt_len, batch, hidden)``.
#             * attns: distribution over src at each tgt
#               ``(tgt_len, batch, src_len)``.
#         """
#
#         dec_state, dec_outs, attns = self._run_forward_pass(
#             tgt, memory_bank, memory_lengths=memory_lengths)
#
#         # Update the state with the result.
#         if not isinstance(dec_state, tuple):
#             dec_state = (dec_state,)
#         self.state["hidden"] = dec_state
#         # self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
#         self.state["coverage"] = None
#         if "coverage" in attns:
#             self.state["coverage"] = attns["coverage"][-1].unsqueeze(0)
#
#         # Concatenates sequence of tensors along a new dimension.
#         # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
#         #       (in particular in case of SRU) it was not raising error in 0.3
#         #       since stack(Variable) was allowed.
#         #       In 0.4, SRU returns a tensor that shouldn't be stacke
#         if type(dec_outs) == list:
#             dec_outs = torch.stack(dec_outs)
#
#             for k in attns:
#                 if type(attns[k]) == list:
#                     attns[k] = torch.stack(attns[k])
#         return dec_outs, attns
