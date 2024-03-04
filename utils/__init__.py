from .config_parser import Parser
from .loss import Loss, LossMultiStage
from .segment_generator import BufferSegment, WindowSegment, WindowSegmentMultiStage, WindowSegmentOneToOneMultiStage
from .statistics import Statistics, StatisticsMultiStage


LOSS = {
    'st-gcn': Loss,
    'co-st-gcn': Loss,
    'rt-st-gcn': Loss,
    'ms-tcn': LossMultiStage,
    'ms-gcn': LossMultiStage,
    'aa-gcn': Loss,
    'shift-gcn': Loss,
    'shift-gcn++': Loss
}

SEGMENT_GENERATOR = {
    'st-gcn': WindowSegment,
    'co-st-gcn': WindowSegment,
    'rt-st-gcn': BufferSegment,
    'ms-tcn': WindowSegmentOneToOneMultiStage,
    'ms-gcn': WindowSegmentMultiStage,
    'aa-gcn': WindowSegment,
    'shift-gcn': WindowSegment,
    'shift-gcn++': WindowSegment
}

STATISTICS = {
    'st-gcn': Statistics,
    'co-st-gcn': Statistics,
    'rt-st-gcn': Statistics,
    'ms-tcn': StatisticsMultiStage,
    'ms-gcn': StatisticsMultiStage,
    'aa-gcn': Statistics,
    'shift-gcn': Statistics,
    'shift-gcn++': Statistics
}
