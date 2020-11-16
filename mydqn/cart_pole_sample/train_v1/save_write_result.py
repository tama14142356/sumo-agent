from torch.utils.tensorboard import SummaryWriter
import os
import torch
import copy

TENSOR_BOARD_LOG_DIR = "results/cartpole/mydqn_result/"
DEVICE_ORIGIN = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SaveWriteResult:
    def __init__(self, episode, demo=False):
        train_or_demo_dir = "train"
        if demo:
            train_or_demo_dir = "demo"
        episode_dir = "episode_{}".format(episode) + "/" + train_or_demo_dir
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
                    ["agent/total_reward", "agent/reward_min", "agent/reward_max"],
                ],
                "reward total_reward loss multiline": [
                    "Multiline",
                    ["agent/reward", "agent/total_reward", "loss"],
                ],
            }
        }
        self.writer.add_custom_scalars(layout)
        self.model_save_filename = os.path.join(self.log_dir, "model.pt")
        self.result_path_log_path = os.path.join(
            os.path.dirname(__file__), "result_path_log.log"
        )
        real_result_path = os.path.abspath(self.log_dir)
        self.is_exist = os.path.isfile(self.result_path_log_path)
        if not demo:
            with open(self.result_path_log_path, "a") as f:
                if self.is_exist:
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
            save_model = copy.deepcopy(model)
            if str(DEVICE_ORIGIN) != device:
                save_model = save_model.to(device)
            file_split_list = os.path.splitext(os.path.basename(filepath))
            file_base_name, file_extension = file_split_list
            save_file_base_name = file_base_name + "_" + str(device)
            save_file_name = save_file_base_name + file_extension
            save_filepath = os.path.join(self.log_dir, save_file_name)
            torch.save(save_model.state_dict(), save_filepath)
        else:
            torch.save(model.state_dict(), filepath)

    def load_model(self, model, filename=""):
        file_path_name = filename
        if len(filename) <= 0:
            file_path_name = self.model_save_filename
        if self.is_exist:
            file_tmp_name = filename
            if len(filename) <= 0:
                file_tmp_name = os.path.basename(self.model_save_filename)
            with open(self.result_path_log_path, "r") as f:
                l_strip = [s.strip() for s in f.readlines()]
                file_path = l_strip[len(l_strip) - 1]
            file_path_name = os.path.join(file_path, file_tmp_name)
        loaded_model = torch.load(file_path_name)
        model.load_state_dict(loaded_model)
        print(model.state_dict())
        model.eval()

    def close(self):
        self.writer.close()
