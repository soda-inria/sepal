import hashlib
import json


# A Control instance stores the configuration information.
class Control:
    def __init__(self):
        self.embed_dim = 100
        self.subgraph_max_size = 4e6
        self.embed_method = "distmult"
        self.embed_setting = DistMultSetting()
        self.seed = 0
        self.core_node_proportions = 0.05
        self.core_selection = "degree"  # "degree" or "hybrid" or "pagerank"
        self.propagation_lr = 1
        self.handle_disconnected = True

    def get_config(self):
        control_params = list(self.__dict__.items())
        embed_params = [
            ("embed_setting." + x[0], x[1]) for x in self.embed_setting.__dict__.items()
        ]

        config = dict(control_params + embed_params)
        config.pop("embed_setting")
        return config


# Embedding methods settings


class HolESetting:
    def __init__(self) -> None:
        self.optimizer = "Adam"
        self.composition = "circular_correlation"
        self.lr = 1e-3


class DistMultSetting:
    def __init__(self) -> None:
        self.optimizer = "Adam"
        self.composition = "multiplication"
        self.lr = 1e-3
        self.loss_fn = "CrossEntropyLoss"


class TransESetting:
    def __init__(self) -> None:
        self.optimizer = "Adam"
        self.composition = "subtraction"
        self.lr = 1e-3


class RotatESetting:
    def __init__(self) -> None:
        self.optimizer = "Adam"
        self.composition = "multiplication"
        self.lr = 1e-3


# List of the methods available for KG embedding.
# If you implement a new method, please add it here, and define a new setting class.
embed_methods = {
    "hole": HolESetting,
    "distmult": DistMultSetting,
    "transe": TransESetting,
    "rotate": RotatESetting,
}


def set_control_params(device, **kwargs):
    """Function to set the parameters for the control instance.

    Args:
        device (str): the device to use, e.g. "cpu" or "cuda:0".
        **kwargs: key-value pairs of parameters to set in the control instance.

    Returns:
        Control: control instance
    """
    # Initialize ctrl
    ctrl = Control()

    # Set attributes
    for key, value in kwargs.items():
        setattr(ctrl, key, value)

    if ctrl.core_selection in ["degree", "pagerank"]:
        ctrl.core_prop = ctrl.core_node_proportions
        ctrl.core_edge_proportions = None
    elif ctrl.core_selection == "hybrid":
        ctrl.core_prop = None

    # Set embedding method's parameters
    ctrl.embed_setting = embed_methods[ctrl.embed_method]()
    if "core_lr" in kwargs:
        ctrl.embed_setting.lr = kwargs["core_lr"]

    # Compute config id
    config = ctrl.get_config()
    ctrl.id = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("ascii")
    ).hexdigest()

    # Set other parameters
    ctrl.device = device
    return ctrl
