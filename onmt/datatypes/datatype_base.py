# coding: utf-8


class Datatype(object):
    def __init__(self, name, reader, sort_key, field_fn):
        self.name = name
        self.reader = reader
        self.sort_key = sort_key
        self.fields = field_fn
