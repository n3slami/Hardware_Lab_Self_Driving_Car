import subprocess
import os
import time


SAVED_MODELS_DIR = "./saved-models"
SAVED_MODEL_PREFIX = "temporary_saved_agent_"
TRAINING_STATS_PREFIX = "training_stats_"


def count_files_with_prefix(list, prefix):
    res = 0
    for name in list:
        if name.startswith(prefix):
            res += 1
    return res


if __name__ == "__main__":
    while True:
        files_list = os.listdir(SAVED_MODELS_DIR)
        model_count = count_files_with_prefix(files_list, SAVED_MODEL_PREFIX)
        stat_count = count_files_with_prefix(files_list, TRAINING_STATS_PREFIX)

        trainer = subprocess.Popen(["python", "-m", "carla_rl.train", "--last_save_counter", f"{model_count}"])
        # trainer = subprocess.Popen(["python", "-m", "carla_rl.train", "--last_save_counter", f"{model_count}",
        #                             "--enable_preview"])
        trainer.wait()
        time.sleep(1)

        try:
            os.rename(f"{SAVED_MODELS_DIR}/{TRAINING_STATS_PREFIX[:-1]}.png",
                    f"{SAVED_MODELS_DIR}/{TRAINING_STATS_PREFIX}{stat_count + 1}.png")
        except FileNotFoundError:
            print("Hmmm, the stats file seems to not be here for some reason... ignoring it")