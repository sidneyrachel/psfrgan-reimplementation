from util.create import create_dataset, create_model
from util.logger import Logger
from util.timer import Timer
from util.config import Config


def train(config):
    dataset = create_dataset(config)
    dataset_size = len(dataset)
    print(f'Number of training images: {dataset_size}.')

    model = create_model(config)
    model.setup(config)

    logger = Logger(config)
    timer = Timer()

    resume_iteration = config.resume_iter_on_top_resume_epoch
    num_iteration_on_one_epoch = dataset_size // config.batch_size
    total_iteration = config.num_epoch * num_iteration_on_one_epoch
    current_iteration = config.resume_epoch * num_iteration_on_one_epoch + resume_iteration
    start_iteration = resume_iteration
    print(f'Start training. Epoch: {config.resume_epoch}. Iteration: {resume_iteration}.')

    for current_epoch in range(config.resume_epoch, config.num_epoch + 1):
        for current_epoch_iteration, data in enumerate(dataset, start=start_iteration):
            current_iteration += 1
            logger.set_current_iteration(current_iteration)

            model.set_input(data, current_iteration)
            timer.update_time('Load data')

            model.forward()
            timer.update_time('Forward')
            model.optimize_parameters()
            loss = model.get_current_losses()
            loss.update(model.get_learning_rate())
            logger.record_losses(loss)
            timer.update_time('Backward')

            if current_iteration % config.print_iteration_frequency == 0:
                print('Iteration progress:')
                epoch_progress_detail = '{:03d}|{:05d}/{:05d}'.format(
                    current_epoch,
                    current_epoch_iteration,
                    num_iteration_on_one_epoch
                )

                logger.print_iteration_summary(
                    epoch_progress_detail=epoch_progress_detail,
                    current_iteration=current_iteration,
                    total_iteration=total_iteration,
                    timer=timer
                )

            if current_epoch_iteration >= num_iteration_on_one_epoch - 1:
                start_iteration = 0
                break

            # TODO: Check if this is true
            #  model.update_learning_rate()

            if config.is_debug:
                break

        print(f'Saving current model at the end of epoch. Epoch: {current_epoch}. Iteration: {current_iteration}.')
        save_prefix = f'epoch_{current_epoch}'
        info = {
            'resume_epoch': current_epoch + 1,
            'resume_iter_on_top_resume_epoch': 0
        }

        model.save_networks(save_prefix, info)

        if config.is_debug and current_epoch >= 0:
            break


if __name__ == '__main__':
    config = Config(
        filename='config/psfrgan/train.json'
    )

    train(config)
