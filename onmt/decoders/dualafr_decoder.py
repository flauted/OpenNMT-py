import torch
import torch.nn as nn

from onmt.decoders.decoder import InputFeedRNNDecoder

try:
    from src.i3dpt import get_padding_shape
except ImportError:
    get_padding_shape = None


class AvgPool3dTFPadding(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(AvgPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
        self.pool = torch.nn.AvgPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class AFRAttentionWeighting(nn.Module):
    def __init__(self, in_feats, intr_feats, hid):
        super(AFRAttentionWeighting, self).__init__()
        self.feat2intr = nn.Linear(in_feats, intr_feats, bias=False)
        self.hid2intr = nn.Linear(hid, intr_feats, bias=False)
        self.intr2one = nn.Linear(intr_feats, 1)
        self.sm = nn.Softmax(dim=0)

    def forward(self, inp, hid):
        return self.sm(self.intr2one(
            self.feat2intr(inp) + self.hid2intr(hid)))


class AFRAttention(nn.Module):
    def __init__(self, in_feats, intr_feats, hid):
        super(AFRAttention, self).__init__()
        self.weighting = AFRAttentionWeighting(in_feats, intr_feats, hid)

    def forward(self, inp, hid):
        weighting = self.weighting(inp, hid)
        return (weighting * inp).sum(0, keepdim=True), weighting


class DualAFRDecoder(InputFeedRNNDecoder):
    def __init__(self, *args, **kwargs):
        super(DualAFRDecoder, self).__init__(*args, **kwargs)
        self.fv2rnn = nn.Linear(1024, self.hidden_size)
        self.feats2rnn = nn.Linear(832, self.hidden_size)
        self.s = AFRAttention(4 * 832, 500, self.hidden_size)
        self.z = AFRAttention(832, 500, self.hidden_size)
        self.U = nn.ModuleList([
            nn.Conv3d(64, 832, (1, 1, 1)),
            nn.Conv3d(192, 832, (1, 1, 1)),
            nn.Conv3d(480, 832, (1, 1, 1))])
        self.rsz = nn.ModuleList([
            torch.nn.Sequential(
                AvgPool3dTFPadding((1, 3, 3), stride=(1, 2, 2)),
                AvgPool3dTFPadding((3, 3, 3), stride=(2, 2, 2)),
                AvgPool3dTFPadding((2, 2, 2), stride=(2, 2, 2))),
            torch.nn.Sequential(
                AvgPool3dTFPadding((3, 3, 3), stride=(2, 2, 2)),
                AvgPool3dTFPadding((2, 2, 2), stride=(2, 2, 2))),
            torch.nn.Sequential(
                AvgPool3dTFPadding((2, 2, 2), stride=(2, 2, 2))),
        ])

    def init_state(self, src, memory_bank, encoder_final):
        """Initialize decoder state with last state of the encoder."""
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        encoder_final = self.fv2rnn(encoder_final)
        self.state["hidden"] = (encoder_final.repeat(self.num_layers, 1, 1),
                                encoder_final.repeat(self.num_layers, 1, 1))

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = \
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        self.state["coverage"] = None

    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        # END Additional args check.

        dec_outs = []
        attns = {"std": []}
        if self.copy_attn is not None or self._reuse_copy_attn:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        dec_state = self.state["hidden"]
        coverage = self.state["coverage"].squeeze(0) \
            if self.state["coverage"] is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for i, mb in enumerate(memory_bank[:-1]):
            memory_bank[i] = self.rsz[i](self.U[i](mb))

        bsz, nF, nz, nx, ny = memory_bank[-1].shape

        feats = torch.cat(memory_bank, dim=1)
        a = feats.view(-1, bsz, nF * len(memory_bank))

        for emb_t in emb.split(1):
            s, p_attn = self.s(a,
                               dec_state[0][0].unsqueeze(0))
            z, _ = self.z(s.view(-1, bsz, nF),
                          dec_state[0][0].unsqueeze(0))
            z = self.feats2rnn(z)
            decoder_input = torch.cat([emb_t.squeeze(0), z.squeeze(0)], 1)
            rnn_output, dec_state = self.rnn(decoder_input,
                                             dec_state)
            decoder_output = rnn_output
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)

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

        dec_state, dec_outs, attns = self._run_forward_pass(
            tgt, memory_bank, memory_lengths=memory_lengths)

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
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
                    attns[k] = torch.stack(attns[k])
        return dec_outs, attns
