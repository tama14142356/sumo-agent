from torch.utils.tensorboard import SummaryWriter
import os
import torch

TENSOR_BOARD_LOG_DIR = "results/sumo-light/mydqn_result/"


class SaveWriteResult:
    def __init__(self, episode):
        episode_dir = "episode_{}".format(episode)
        log_dir = os.path.join(TENSOR_BOARD_LOG_DIR, episode_dir)
        log_num, is_exist = 0, True
        while is_exist:
            tmp_dir = os.path.join(log_dir, "{}".format(log_num))
            is_exist = os.path.exists(tmp_dir)
            log_num += 1
        self.log_dir = tmp_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        layout = {
            "Aggregate Charts": {
                "total_reward margin": [
                    "Margin",
                    ["total_reward", "reward_min", "reward_max"],
                ],
                "reward total_reward loss multiline": [
                    "Multiline",
                    ["reward", "total_reward", "loss"],
                ],
            }
        }
        self.writer.add_custom_scalars(layout)
        self.model_save_filename = os.path.join(self.log_dir, "model.pt")
        self.result_path_log_path = os.path.join(
            os.path.dirname(__file__), "result_path_log.log"
        )
        real_result_path = os.path.abspath(self.log_dir)
        is_exist = os.path.isfile(self.result_path_log_path)
        with open(self.result_path_log_path, "a") as f:
            if is_exist:
                print("", file=f)
            print(real_result_path, file=f)

    def writing_list(self, tag, target_list, end_step):
        scalar_num = len(target_list)
        start_step = end_step + 1 - scalar_num
        for step, scalar in zip(range(start_step, end_step + 1), target_list):
            self.writer.add_scalar(tag=tag, scalar_value=scalar, global_step=step)

    def save_model(self, model, filename="", device=None):
        if len(filename) <= 0:
            filepath = self.model_save_filename
        else:
            filepath = os.path.join(self.log_dir, filename)
        if device is not None:
            save_model = model.to(device)
            file_split_list = os.path.splitext(os.path.basename(filepath))
            file_base_name, file_extension = file_split_list
            save_file_base_name = file_base_name + "_" + str(device)
            save_file_name = save_file_base_name + file_extension
            save_filepath = os.path.join(self.log_dir, save_file_name)
            torch.save(save_model.state_dict(), save_filepath)
        else:
            torch.save(model.state_dict(), filepath)

    def load_model(self, model, filename=""):
        if len(filename) <= 0:
            filename = self.model_save_filename
        loaded_model = torch.load(filename)
        model.load_state_dict(loaded_model.state_dict())

    def close(self):
        self.writer.close()
