from .fql_agent import FQLAgent
from .sacbc_agent import SACBCAgent
from .qsm_agent import QSMAgent
from .dsrl_agent import DSRLAgent
from .ifql_agent import IFQLAgent
from .custom_agent import CustomAgent

agents = {
    "fql": FQLAgent,
    "sacbc": SACBCAgent,
    "qsm": QSMAgent,
    "dsrl": DSRLAgent,
    "ifql": IFQLAgent,
    "custom": CustomAgent,
}
