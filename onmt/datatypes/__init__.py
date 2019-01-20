# coding: utf-8

from onmt.datatypes.datatype_base import Datatype
from onmt.datatypes.reader_base import DatasetReaderBase
from onmt.datatypes.audio_datatype import audio_datatype
from onmt.datatypes.image_datatype import image_datatype
from onmt.datatypes.text_datatype import text_datatype

str2datatype = {
    audio_datatype.name: audio_datatype,
    image_datatype.name: image_datatype,
    text_datatype.name: text_datatype
}

__all__ = ["Datatype", "DatasetReaderBase", "audio_datatype",
           "image_datatype", "text_datatype", "str2datatype"]
