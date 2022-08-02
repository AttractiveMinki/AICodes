import torch
import numpy as np
import random

"""
pytorch에서 seed 값을 고정할 때 쓰입니다.

다음과 같이 불러와서 사용할 수 있습니다.
from data_utils import seed_everything

seed_everything(42)


argparse.ArgumentParser()등을 이용해 다음과 같이 사용할 수도 있습니다.

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
seed_everything(args.seed)
"""
def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


"""
Pytorch에서 그래핔 카드를 사용할 수 있으면 그래픽카드를,
사용할 수 없다면 cpu를 사용합니다.
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

