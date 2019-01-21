# coding: utf-8

from itertools import chain
from collections import Counter

import torch
from torchtext.data import Example
from torchtext.data import Dataset as TorchtextDataset
from torchtext.vocab import Vocab


class Dataset(TorchtextDataset):
    """
    A dataset is an object that accepts sequences of raw data (sentence pairs
    in the case of machine translation) and fields which describe how this
    raw data should be processed to produce tensors. When a dataset is
    instantiated, it applies the fields' preprocessing pipeline (but not
    the bit that numericalizes it or turns it into batch tensors) to the raw
    data, producing a list of torchtext.data.Example objects. torchtext's
    iterators then know how to use these examples to make batches.

    Args:
        src_data_type (onmt.datatypes.Datatype)
        tgt_data_type (onmt.datatypes.Datatype)
        fields (dict): a dict with the structure returned by
            inputters.get_fields(). keys match the keys of items
            yielded by the src_examples_iter or tgt_examples_iter,
            while values are lists of (name, Field) pairs.
            An attribute with this name will be created for each
            Example object, and its value will be the result of
            applying the Field to the data that matches the key.
            The advantage of having sequences of fields
            for each piece of raw input is that it allows for
            the dataset to store multiple `views` of each input,
            which allows for easy implementation of token-level
            features, mixed word- and character-level models, and so on.
        src_examples_iter (Sequence[dict]): Each dict's keys should be a
            subset of the keys in `fields`.
        tgt_examples_iter (Sequence[dict] or NoneType): like
            `src_examples_iter`, but may be None (this is the case at
            translation time if no target is specified).
        filter_pred (Callable[[Example], bool]):
            A function that accepts Example objects and returns a
            boolean value indicating whether to include that example
            in the dataset.

    Attributes:
        examples (list[torchtext.data.Example]): a list of
            `torchtext.data.Example` objects with attributes as
            described above (inherited).
        fields (dict[str, torchtext.data.Field]): keys are
            strings with the same names as the
            attributes of the elements of `examples` and whose values are
            the corresponding `torchtext.data.Field` objects. NOTE: this is not
            the same structure as in the fields argument passed to the
            constructor.
        src_vocabs (list[Vocab])
        src_data_type (onmt.datatypes.Datatype)
        tgt_data_type (onmt.datatypes.Datatype)

    """

    def __init__(self, src_data_type, tgt_data_type, fields, src_examples_iter,
                 tgt_examples_iter, filter_pred=None):
        self.src_data_type = src_data_type
        self.sort_key = src_data_type.sort_key
        self.tgt_data_type = tgt_data_type

        dynamic_dict = 'src_map' in fields and 'alignment' in fields

        if tgt_examples_iter is not None:
            examples_iter = (self._join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))
        else:
            examples_iter = src_examples_iter

        # self.src_vocabs is used in collapse_copy_scores and Translator.py
        self.src_vocabs = []
        examples = []
        for ex_dict in examples_iter:
            if dynamic_dict:
                src_field = fields['src'][0][1]
                tgt_field = fields['tgt'][0][1]
                src_vocab, ex_dict = self._dynamic_dict(
                    ex_dict, src_field, tgt_field)
                self.src_vocabs.append(src_vocab)
            ex_fields = {k: v for k, v in fields.items() if k in ex_dict}
            ex = Example.fromdict(ex_dict, ex_fields)
            examples.append(ex)

        # the dataset's self.fields should have the same attributes as examples
        fields = dict(chain.from_iterable(ex_fields.values()))

        super(Dataset, self).__init__(examples, fields, filter_pred)

    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if 'fields' not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)

    def _join_dicts(self, *args):
        """
        Args:
            dictionaries with disjoint keys.

        Returns:
            a single dictionary that has the union of these keys.
        """
        return dict(chain(*[d.items() for d in args]))

    def _dynamic_dict(self, example, src_field, tgt_field):
        src = src_field.tokenize(example["src"])
        # make a small vocab containing just the tokens in the source sequence
        unk = src_field.unk_token
        pad = src_field.pad_token
        src_vocab = Vocab(Counter(src), specials=[unk, pad])
        # Map source tokens to indices in the dynamic dict.
        src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
        example["src_map"] = src_map

        if "tgt" in example:
            tgt = tgt_field.tokenize(example["tgt"])
            mask = torch.LongTensor(
                [0] + [src_vocab.stoi[w] for w in tgt] + [0])
            example["alignment"] = mask
        return src_vocab, example

    @property
    def can_copy(self):
        return self.data_type == 'text' and (
            "src_map" in self.fields and "alignment" in self.fields)

    @staticmethod
    def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs,
                             batch_dim=1, batch_offset=None):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambiguous.
        """
        offset = len(tgt_vocab)
        for b in range(scores.size(batch_dim)):
            blank = []
            fill = []
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = batch.indices.data[batch_id]
            src_vocab = src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    blank.append(offset + i)
                    fill.append(ti)
            if blank:
                blank = torch.Tensor(blank).type_as(batch.indices.data)
                fill = torch.Tensor(fill).type_as(batch.indices.data)
                score = scores[:, b] if batch_dim == 1 else scores[b]
                score.index_add_(1, fill, score.index_select(1, blank))
                score.index_fill_(1, blank, 1e-10)
        return scores
