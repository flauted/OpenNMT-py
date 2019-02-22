"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder
from onmt.decoders.transformer import TransformerDecoder
from onmt.decoders.cnn_decoder import CNNDecoder
from onmt.decoders.dualafr_decoder import DualAFRDecoder


str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder,
           "cnn": CNNDecoder, "transformer": TransformerDecoder,
           "dualafr": DualAFRDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder", "CNNDecoder",
           "InputFeedRNNDecoder", "DualAFRDecoder", "str2dec"]
