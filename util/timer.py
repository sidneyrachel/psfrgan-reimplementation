import time
import datetime
from collections import OrderedDict


class Timer():
    def __init__(self):
        self.reset_timer()
        self.start = time.time()

    def reset_timer(self):
        self.before = time.time()
        self.timer_map = OrderedDict()

    def restart(self):
        self.before = time.time()

    def update_time(self, key):
        self.timer_map[key] = time.time() - self.before
        self.before = time.time()

    def to_string(self, iteration_left, is_short=False):
        duration_for_current_iteration = sum(self.timer_map.values())
        message = '{:%Y-%m-%d %H:%M:%S}\tElapsed time: {}\tTime left: {}\t'.format(
            datetime.datetime.now(),
            datetime.timedelta(seconds=round(time.time() - self.start)),
            datetime.timedelta(seconds=round(duration_for_current_iteration * iteration_left))
        )
        if is_short:
            message += '{}: {:.2f}s'.format('|'.join(self.timer_map.keys()), duration_for_current_iteration)
        else:
            message += 'Cur iter total duration: {:.2f}s\t{}: {}'.format(
                duration_for_current_iteration,
                '|'.join(self.timer_map.keys()),
                ' '.join('{:.2f}s'.format(value) for value in self.timer_map.values())
            )

        return message
