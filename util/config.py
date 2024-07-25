from dataclasses import dataclass


@dataclass
class GraphConfig:
    DEGREE: int = 2
    GRAPH_SEED: int = 665
    REAL_GRAPH: bool = False


@dataclass
class ServerConfig:
    HOP_NUM: int = 3
    ELEMENT_NUM: int = 1000
    CAPACITY_LIST: list = 150, 175, 200, 225
    CAPACITY_MIN: int = 350
    CAPACITY_MAX: int = 700
    CACHE_PERCENTAGE: float = 0.25
    CONNECTION_POOL_SIZE: int = 5
    CACHE_TYPE: str = 'CLOUD'



@dataclass
class NetWorkConfig:
    ONE_HOP_TIME: int = 10  # ms
    TIME_OUT: int = 200  # ms
    CLOUD_HOP = 20


@dataclass
class TestConfig:
    NODENUM: int = 10
    DATANUM: int = 8000
    CACHE_HOT_RATE: float = 0.6
    MAX_DEDUPLICATE_RATE: float = 1
    DIS_DEDUP_RATE: float = 0.0
    ALPHA: float = 0.7
    # 0 : global deduplication, 1: distributed deduplication basic
    # 2:distributed deduplication dependency 3: BEDD
    # 4:Random  5:real_random 6:MEAN 7:LDI 8:CDI
    DEDUPLICATE_STRATEGY: int = 2
    DEDUPLICATE_MOD: int = 1  # 0 for simple mod and 1 for dependency mod
    SIMULATION: bool = True
    REGISTER_DEPENDENCY: bool = True
    CLOSE_MESS_SHOW: bool = True
    DATA_QUERY_SHOW: bool = True
    DATA_CACHE_SHOW: bool = False
    HBFC_UPDATE_SHOW: bool = False
    DATA_HEAT_SHOW: bool = False
    DATA_DEDUP_SHOW: bool = False
    DATA_PREDELETE_SHOW: bool = False
    DELETE_PERMI_SHOW: bool = False
    DATA_CHECK_SHOW: bool = False
    REGISTER_SHOW: bool = False
    IGNORE_HOT: bool = False


