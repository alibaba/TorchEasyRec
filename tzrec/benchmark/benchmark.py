# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

from tzrec.utils import config_util, misc_util

TEXT_RESET = "\033[0m"
TEXT_BOLD_RED = "\033[1;31m"
TEXT_BOLD_GREEN = "\033[1;32m"
TEXT_BOLD_YELLOW = "\033[1;33m"
TEXT_BOLD_BLUE = "\033[1;34m"
TEXT_BOLD_CYAN = "\033[1;36m"


# pyre-ignore [2,3]
def print_error(*args, **kwargs):
    """Print error info."""
    print(f"{TEXT_BOLD_RED}[ERROR]{TEXT_RESET}", *args, **kwargs)


# pyre-ignore [2,3]
def print_worse(*args, **kwargs):
    """Print train eval metric all worse info."""
    print(f"{TEXT_BOLD_BLUE}[WORSE]{TEXT_RESET}", *args, **kwargs)


# pyre-ignore [2,3]
def print_better(*args, **kwargs):
    """Print train eval metric all better info."""
    print(f"{TEXT_BOLD_GREEN}[BETTER]{TEXT_RESET}", *args, **kwargs)


# pyre-ignore [2,3]
def print_some_better_and_worse(*args, **kwargs):
    """Has some better and worse metric info."""
    print(f"{TEXT_BOLD_CYAN}[HAS BETTER AND WORSE]{TEXT_RESET}", *args, **kwargs)


# pyre-ignore [2,3]
def print_balance(*args, **kwargs):
    """Train metric not much change."""
    print(f"{TEXT_BOLD_YELLOW}[BALANCE]{TEXT_RESET}", *args, **kwargs)


def _get_benchmark_project() -> str:
    """Get ODPS project for benchmark."""
    project = os.environ.get("CI_ODPS_PROJECT_NAME", "")
    if "ODPS_CONFIG_FILE_PATH" in os.environ:
        with open(os.environ["ODPS_CONFIG_FILE_PATH"], "r") as f:
            for line in f.readlines():
                values = line.split("=", 1)
                if len(values) == 2 and values[0] == "project_name":
                    project = values[1].strip()
    return project


def _modify_pipline_config(
    pipeline_config_path: str,
    model_path: str,
    run_config_path: str,
) -> None:
    pipeline_config = config_util.load_pipeline_config(pipeline_config_path)
    pipeline_config.model_dir = model_path
    project = _get_benchmark_project()
    train_input_path = pipeline_config.train_input_path.format(PROJECT=project)
    pipeline_config.train_input_path = train_input_path
    eval_input_path = pipeline_config.eval_input_path.format(PROJECT=project)
    pipeline_config.eval_input_path = eval_input_path

    if pipeline_config.data_config.HasField("negative_sampler"):
        sampler = pipeline_config.data_config.negative_sampler
        sampler.input_path = sampler.input_path.format(PROJECT=project)
    elif pipeline_config.data_config.HasField("negative_sampler_v2"):
        sampler = pipeline_config.data_config.negative_sampler_v2
        sampler.user_input_path = sampler.user_input_path.format(PROJECT=project)
        sampler.item_input_path = sampler.item_input_path.format(PROJECT=project)
        sampler.pos_edge_input_path = sampler.pos_edge_input_path.format(
            PROJECT=project
        )
    elif pipeline_config.data_config.HasField("hard_negative_sampler"):
        sampler = pipeline_config.data_config.hard_negative_sampler
        sampler.user_input_path = sampler.user_input_path.format(PROJECT=project)
        sampler.item_input_path = sampler.item_input_path.format(PROJECT=project)
        sampler.hard_neg_edge_input_path = sampler.hard_neg_edge_input_path.format(
            PROJECT=project
        )
    elif pipeline_config.data_config.HasField("hard_negative_sampler_v2"):
        sampler = pipeline_config.data_config.hard_negative_sampler_v2
        sampler.user_input_path = sampler.user_input_path.format(PROJECT=project)
        sampler.item_input_path = sampler.item_input_path.format(PROJECT=project)
        sampler.pos_edge_input_path = sampler.pos_edge_input_path.format(
            PROJECT=project
        )
        sampler.hard_neg_edge_input_path = sampler.hard_neg_edge_input_path.format(
            PROJECT=project
        )
    elif pipeline_config.data_config.HasField("tdm_sampler"):
        sampler = pipeline_config.data_config.tdm_sampler
        sampler.item_input_path = sampler.item_input_path.format(PROJECT=project)
        sampler.edge_input_path = sampler.edge_input_path.format(PROJECT=project)
        sampler.predict_edge_input_path = sampler.predict_edge_input_path.format(
            PROJECT=project
        )
    config_util.save_message(pipeline_config, run_config_path)


def _benchmark_train_eval(
    run_config_path: str,
    log_path: str,
) -> bool:
    """Run train_eval for benchmark."""
    cmd_str = (
        "PYTHONPATH=. torchrun --standalone "
        "--nnodes=1 --nproc-per-node=2 "
        f"--log_dir {log_path} -r 3 -t 3 tzrec/train_eval.py "
        f"--pipeline_config_path {run_config_path}"
    )
    return misc_util.run_cmd(cmd_str, log_path + ".log", timeout=6000)


def _get_config_paths(pipeline_config_paths: str) -> List[str]:
    """Get dir all pipeline config path."""
    config_paths = []
    if os.path.isfile(pipeline_config_paths):
        config_paths.append(pipeline_config_paths)
    elif os.path.isdir(pipeline_config_paths):
        for root, _, files in os.walk(pipeline_config_paths):
            for file in files:
                if "base_eval_metric.json" != file:
                    config_paths.append(os.path.join(root, file))
    else:
        raise Exception(f"{pipeline_config_paths} is not a valid file or directory")
    return config_paths


def _create_directory(path: str) -> str:
    """Create the directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def _get_train_metrics(path: str) -> Dict[str, Any]:
    """From model path we get eval metrics."""
    eval_file = os.path.join(path, "train_eval_result.txt")
    f = open(eval_file)
    metrics = json.load(f)
    return metrics


def _compare_metrics(
    metric_config: Dict[str, Any], train_metrics: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compare model metrics and base metrics."""
    base_metric = metric_config["label"]
    threshold = metric_config["threshold"]
    train_avg_metric = {}
    change_metric = {}
    better_name = []
    worse_name = []
    balance_name = []
    for k, v in base_metric.items():
        if isinstance(threshold, dict):
            task_threshold = threshold[k]
        else:
            task_threshold = threshold
        if len(train_metrics) > 0:
            train_avg_v = sum([metric[k] for metric in train_metrics]) / len(
                train_metrics
            )
            train_avg_metric[k] = train_avg_v
            if train_avg_v - v >= task_threshold:
                better_name.append(k)
            elif train_avg_v - v <= -task_threshold:
                worse_name.append(k)
            else:
                balance_name.append(k)
    change_metric["better"] = better_name
    change_metric["worse"] = worse_name
    change_metric["balance"] = balance_name
    return train_avg_metric, change_metric


def _print(
    config_path: str,
    run_cnt: int,
    fail_cnt: int,
    train_avg_metric: Dict[str, Any],
    change_metric: Dict[str, Any],
) -> None:
    """Print train metrics."""
    better_name = change_metric["better"]
    worse_name = change_metric["worse"]
    balance_name = change_metric["balance"]
    success_cnt = run_cnt - fail_cnt
    msg = f"config_path:{config_path}, fail_cnt:{fail_cnt} and run_cnt:{success_cnt}. "
    if fail_cnt >= run_cnt - fail_cnt:
        msg = msg + f"metric is unbelief, train metric: {train_avg_metric}"
        print_error(msg)
    elif len(better_name) > 0 and len(worse_name) == 0:
        msg = (
            msg + f"better metric name:{better_name}, "
            f"all train metric is:{train_avg_metric}"
        )

        print_better(msg)
    elif len(worse_name) > 0 and len(better_name) == 0:
        msg = (
            msg
            + f"worse metric name:{worse_name}, all train metric is:{train_avg_metric}"
        )
        print_worse(msg)
    elif len(worse_name) > 0 and len(better_name) > 0:
        msg = (
            msg + f"worse metric name:{better_name}, better metric name:{better_name}, "
            f"all train metric is:{train_avg_metric}"
        )
        print_some_better_and_worse(msg)
    elif len(balance_name) > 0 and len(worse_name) == 0 and len(better_name) == 0:
        msg = (
            msg + f"not has better and worse metric name, "
            f"all train metric is:{train_avg_metric}"
        )
        print_balance(msg)
    else:
        msg = msg + f"all train metric is:{train_avg_metric}"
        print(msg)


def main(
    pipeline_config_paths: str,
    experiment_path: str,
    base_metric_path: str = "tzrec/benchmark/configs/base_eval_metric.json",
) -> None:
    """Run benchmarks."""
    train_config_paths = _get_config_paths(pipeline_config_paths)
    f = open(base_metric_path)
    base_eval_metrics = json.load(f)
    experiment_path = experiment_path + f"_{int(time.time())}"
    print(f"******* We will save experiment is {experiment_path} *******")
    models_path = _create_directory(os.path.join(experiment_path, "models"))
    configs_path = _create_directory(os.path.join(experiment_path, "configs"))
    logs_path = _create_directory(os.path.join(experiment_path, "logs"))

    all_train_metrics = {}
    all_train_metrics_info = {}
    for old_config_path in train_config_paths:
        metric_config = base_eval_metrics[old_config_path]
        run_cnt = metric_config["run_cnt"]
        train_metrics = []
        fail_cnt = 0
        for i in range(run_cnt):
            file_path = (
                old_config_path.replace("/", "_")
                .replace("\\", "_")
                .replace(".config", "")
            )
            file_path = file_path + f"_{i}"
            new_config_path = os.path.join(configs_path, file_path + ".config")
            model_path = os.path.join(models_path, file_path)
            log_path = os.path.join(logs_path, file_path)
            _modify_pipline_config(old_config_path, model_path, new_config_path)
            success = _benchmark_train_eval(new_config_path, log_path)
            if success:
                train_metric = _get_train_metrics(model_path)
                train_metrics.append(train_metric)
            else:
                fail_cnt += 1
        train_avg_metric, change_metric = _compare_metrics(metric_config, train_metrics)

        _print(old_config_path, run_cnt, fail_cnt, train_avg_metric, change_metric)
        all_train_metrics[old_config_path] = train_avg_metric
        print_info = {
            "run_cnt": run_cnt,
            "fail_cnt": fail_cnt,
            "train_avg_metric": train_avg_metric,
            "change_metric": change_metric,
        }
        all_train_metrics_info[old_config_path] = print_info
    print("".join(["="] * 30))
    print("".join(["="] * 30))
    for old_config_path, print_info in all_train_metrics_info.items():
        run_cnt = print_info["run_cnt"]
        fail_cnt = print_info["fail_cnt"]
        train_avg_metric = print_info["train_avg_metric"]
        change_metric = print_info["change_metric"]
        _print(old_config_path, run_cnt, fail_cnt, train_avg_metric, change_metric)
    benchmark_file = os.path.join(experiment_path, "benchmark_eval.txt")
    with open(benchmark_file, "w") as f:
        json.dump(all_train_metrics, f)
    print("benchmark complete !!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_config_path",
        type=str,
        default="tzrec/benchmark/configs",
        help="Path to pipeline config file.",
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        default="tmp",
        help="Path to experiment model save.",
    )
    args, extra_args = parser.parse_known_args()
    main(
        args.pipeline_config_path,
        args.experiment_path,
    )
