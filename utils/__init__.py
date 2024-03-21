from .config_parser import Parser
from .loss import Loss, LossMultiStage, LossOneToOneMultiStage
from .segment_generator import BufferSegment, WindowSegment, WindowSegmentMultiStage, WindowSegmentOneToOneMultiStage
from .statistics import Statistics, StatisticsMultiStage, StatisticsOneToOneMultiStage


LOSS = {
    'st-gcn': Loss,
    'rt-st-gcn': Loss,
    'ms-tcn': LossOneToOneMultiStage,
    'ms-gcn': LossMultiStage,
    'aa-gcn': Loss,
    'shift-gcn': Loss,
    'shift-gcn++': Loss
}

SEGMENT_GENERATOR = {
    'st-gcn': WindowSegment,
    'rt-st-gcn': BufferSegment,
    'ms-tcn': WindowSegmentOneToOneMultiStage,
    'ms-gcn': WindowSegmentMultiStage,
    'aa-gcn': WindowSegment,
    'shift-gcn': BufferSegment,
    'shift-gcn++': WindowSegment
}

STATISTICS = {
    'st-gcn': Statistics,
    'rt-st-gcn': Statistics,
    'ms-tcn': StatisticsOneToOneMultiStage,
    'ms-gcn': StatisticsMultiStage,
    'aa-gcn': Statistics,
    'shift-gcn': Statistics,
    'shift-gcn++': Statistics
}
