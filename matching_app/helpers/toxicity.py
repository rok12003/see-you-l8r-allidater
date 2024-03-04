

import pandas as pd
import numpy as np

# Toxic-detector/score package:
from detoxify import Detoxify

# Saving original Detoxify model here as well:
model = Detoxify("original")

def detox(bio):
    return model.predict(bio)