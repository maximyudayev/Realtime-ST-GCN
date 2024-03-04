from .rtstgcn import Model as RtStgcn
from .stgcn import Model as Stgcn
from .costgcn import Model as CoStgcn
from .mstcn import Model as MsTcn
from .msgcn import Model as MsGcn
from .aagcn import Model as AaGcn
# from .shiftgcn import Model as ShiftGcn
# from .shiftgcn_plus import Model as ShiftGcnPlus


MODELS = {
    'st-gcn': Stgcn,
    'co-st-gcn': CoStgcn,
    'rt-st-gcn': RtStgcn,
    'ms-tcn': MsTcn,
    'ms-gcn': MsGcn,
    'aa-gcn': AaGcn,
    # 'shift-gcn': ShiftGcn,
    # 'shift-gcn++': ShiftGcnPlus
}
