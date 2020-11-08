from torch.utils.tensorboard import SummaryWriter
import os
import torch

TENSOR_BOARD_LOG_DIR = 'results/sumo-light1/mydqn/episode_10'


class SaveWriteResult:
    def __init__(self):
        self.writer = SummaryWriter(log_dir=TENSOR_BOARD_LOG_DIR)
        self.model_save_filename = os.path.join(
            TENSOR_BOARD_LOG_DIR, "model.pt")
        result_path_log_path = os.path.join(
            os.path.dirname(__file__), "result_path_log.log")
        real_result_path = os.path.join(os.getcwd(), TENSOR_BOARD_LOG_DIR)
        with open(result_path_log_path, 'a') as f:
            print(real_result_path, file=f)

    def writing_list(self, tag, target_list, end_step):
        scalar_num = len(target_list)
        start_step = end_step + 1 - scalar_num
        for step, scalar in zip(range(start_step, end_step + 1), target_list):
            self.writer.add_scalar(
                tag=tag, scalar_value=scalar, global_step=step)

    def save_model(self, model, filename="", device=None):
        if len(filename) <= 0:
            filepath = self.model_save_filename
        else:
            filepath = os.path.join(TENSOR_BOARD_LOG_DIR, filename)
        if device is not None:
            save_model = model.to(device)
            torch.save(save_model.state_dict(), filepath)
        else:
            torch.save(model.state_dict(), filepath)

    def load_model(self, model, filename=""):
        if len(filename) <= 0:
            filename = self.model_save_filename
        loaded_model = torch.load(filename)
        model.load_state_dict(loaded_model.state_dict())

    def close(self):
        self.writer.close()
