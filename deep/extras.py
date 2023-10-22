from imports import *


def seed_everything(seed):
    """
    Sets the seed for random, numpy and torch and torch.cuda.

    Parameters:
        seed (int): The seed value to set for the random number generators.

    Returns:
        None
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.backends.cudnn.benchmark = False
    t.use_deterministic_algorithms(True)
