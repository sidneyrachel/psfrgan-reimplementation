from variable.config import init_config
from variable import config as cfg_holder
from dataset.util import make_dataset


if __name__ == '__main__':
    init_config('config/test.json')
    print(cfg_holder.config.mask_size)

    a = make_dataset('sample/imgs1024', filename_set=set())
    b = make_dataset('sample/masks512')
    print(a)
    print(b)
