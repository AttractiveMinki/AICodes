import torch
import numpy as np
import random

"""
pytorch에서 seed 값을 고정할 때 쓰입니다.

다음과 같이 불러와서 사용할 수 있습니다.
from data_utils import seed_everything

seed_everything(42)
42 자리에 argparse.ArgumentParser()등을 이용해 args.seed를 넣기도 합니다.
"""
def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)