import os
import random
import shutil
import sys
import time
from collections import defaultdict
from typing import Annotated, Any

import einops as ein
import lightning.pytorch as pl  # type: ignore
import matplotlib.pyplot as plt
import neptune  # type: ignore
import numpy as np
import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype as typed
from beartype.door import die_if_unbearable as assert_type
from beartype.typing import Callable, Iterable
from beartype.vale import Is
from datasets import load_dataset, load_from_disk  # type: ignore
from jaxtyping import Float, Int
from lightning.pytorch.callbacks import LearningRateMonitor  # type: ignore
from lightning.pytorch.loggers import CSVLogger, NeptuneLogger  # type: ignore
from numpy import ndarray as ND
from torch import Tensor as TT
from tqdm import tqdm
