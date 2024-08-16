from bdrate import BDrate
from bd_rate_np import BD_RATE
import torch
    
bd_torch = BDrate()
bd_np = BD_RATE
args = ([686.76, 309.58, 157.11, 85.95],
            [40.28, 37.18, 34.24, 31.42],
            [893.34, 407.8, 204.93, 112.75],
            [40.39, 37.21, 34.17, 31.24])

bd_np_val = bd_np(*args)
args = [torch.tensor(arg).double() for arg in args]
diff = (bd_torch(*args) - bd_np_val / 100) / (bd_np_val / 100)

print(diff)
    
