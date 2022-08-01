import os
from datetime import datetime
from collections import OrderedDict
import shutil

from util.common import make_directories
from variable.model import PhaseEnum


class Logger():
    def __init__(self, config):
        timestamp = '{}'.format(datetime.now().strftime('%Y-%m-%d_%H:%M'))
        self.config = config
        self.log_directory = os.path.join(config.log_directory, f'{config.experiment_name}_{timestamp}')
        self.phase_keys = [PhaseEnum.TRAIN.value, PhaseEnum.VAL.value, PhaseEnum.TEST.value]
        self.iteration_logs = []
        self.set_phase(config.phase)

        existing_log = None

        make_directories([
            config.log_directory,
            config.log_archive_directory
        ])

        for log_name in os.listdir(config.log_directory):
            if config.experiment_name in log_name:
                existing_log = log_name

        if existing_log is not None:
            old_directory = os.path.join(config.log_directory, existing_log)
            archive_directory = os.path.join(config.log_archive_directory, existing_log)
            shutil.move(old_directory, archive_directory)

        make_directories(self.log_directory)

        self.make_log_file()

    def make_log_file(self):
        self.text_files = OrderedDict()

        for phase in self.phase_keys:
            self.text_files[phase] = os.path.join(self.log_directory, f'log_{phase}')

    def set_phase(self, phase):
        self.phase = phase

    def set_current_iteration(self, current_iteration):
        self.current_iteration = current_iteration

    def record_losses(self, items):
        self.iteration_logs.append(items)

    def print_iteration_summary(
            self,
            epoch_progress_detail,
            current_iteration,
            total_iteration,
            timer
    ):
        message = '{}\nIter: [{}]{:03d}/{:03d}\t\t'.format(
            timer.to_string(total_iteration - current_iteration),
            epoch_progress_detail,
            current_iteration,
            total_iteration
        )

        for key, value in self.iteration_logs[-1].items():
            message += '{}: {:.6f}\t'.format(key, value)

        print(f'{message}\n')

        with open(self.text_files[self.phase], 'a+') as f:
            f.write(f'{message}\n')
