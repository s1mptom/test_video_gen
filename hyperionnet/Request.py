# automatically generated by the FlatBuffers compiler, do not modify

# namespace: hyperionnet

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Request(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Request()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsRequest(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Request
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Request
    def CommandType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # Request
    def Command(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

def RequestStart(builder):
    builder.StartObject(2)

def Start(builder):
    RequestStart(builder)

def RequestAddCommandType(builder, commandType):
    builder.PrependUint8Slot(0, commandType, 0)

def AddCommandType(builder, commandType):
    RequestAddCommandType(builder, commandType)

def RequestAddCommand(builder, command):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(command), 0)

def AddCommand(builder, command):
    RequestAddCommand(builder, command)

def RequestEnd(builder):
    return builder.EndObject()

def End(builder):
    return RequestEnd(builder)
