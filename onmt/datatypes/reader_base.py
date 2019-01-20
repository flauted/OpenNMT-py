# coding: utf-8

import codecs


class MissingDependencyException(Exception):
    pass


def missing_dependency_exception(reader_name, *args):
    return MissingDependencyException(
        "Could not create reader {:s}. Be sure to install "
        "the following dependencies: ".format(reader_name) + ", ".join(args)
    )


class DatasetReaderBase(object):
    @classmethod
    def from_opt(cls, opt):
        return cls()

    def read(self, src, side, src_dir=None):
        raise NotImplementedError()

    @staticmethod
    def _read_file(path):
        with codecs.open(path, "r", "utf-8") as f:
            for line in f:
                yield line
