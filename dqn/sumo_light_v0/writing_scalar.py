from torch.utils.tensorboard import SummaryWriter

TENSOR_BOARD_LOG_DIR = './results/sumo-light1'


class WritingScalar:
    def __init__(self):
        self.writer = SummaryWriter(log_dir=TENSOR_BOARD_LOG_DIR)

    def writing_list(self, tag, target_list, end_step):
        scalar_num = len(target_list)
        start_step = end_step + 1 - scalar_num
        for step, scalar in zip(range(start_step, end_step + 1), target_list):
            self.writer.add_scalar(
                tag=tag, scalar_value=scalar, global_step=step)

    def close(self):
        self.writer.close()
