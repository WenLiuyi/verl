# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Union

import subprocess
import os
import signal
import time

class Tracking:
    supported_backend = ["wandb", "mlflow", "swanlab", "vemlp_wandb", "tensorboard", "console", "rlloggingboard"]

    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = "console", config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}

        if "tracking" in default_backend or "wandb" in default_backend:
            import wandb

            wandb.init(project=project_name, name=experiment_name, config=config)
            self.logger["wandb"] = wandb

        if "mlflow" in default_backend:
            import os

            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", None)
            if MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten
            
            if config is None:
                config = {} # make sure config is not None, otherwise **config will raise error
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import os

            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter()

        if "console" in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "rlloggingboard" in default_backend:
            self.logger["rlloggingboard"] = _RLLoggingboardAdapter(
                config.trainer.rl_logging_board_dir, 
                project_name, 
                experiment_name)
            rl_backend=_RLLoggingBoardBackend()
            rl_backend.start()


    def log(self, data, step, batch, backend=None, tokenizer=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step, batch=batch, tokenizer=tokenizer)

    def __del__(self):
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)
        if "swanlab" in self.logger:
            self.logger["swanlab"].finish()
        if "vemlp_wandb" in self.logger:
            self.logger["vemlp_wandb"].finish(exit_code=0)
        if "tensorboard" in self.logger:
            self.logger["tensorboard"].finish()


class _TensorboardAdapter:
    def __init__(self):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", "tensorboard_log")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step, batch=None):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def log(self, data, step):
        import mlflow

        results = {k.replace("@", "_at_"): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:
    def log(self, loggers, samples, step):
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)

    def log_generations_to_wandb(self, samples, step):
        """Log samples to wandb as a table"""
        import wandb

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], [])

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_text_list = []
        for i, sample in enumerate(samples):
            row_text = f"""
            input: {sample[0]}
            
            ---
            
            output: {sample[1]}
            
            ---
            
            score: {sample[2]}
            """
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i + 1}"))

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_text_list}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import json
        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

class _RLLoggingboardAdapter:
    from verl import DataProto
    def __init__(
        self,
        root_log_dir: str,
        project_name: str,
        experiment_name: str
    ):
        self.save_path = os.path.join(
            root_log_dir, 
            project_name, 
            experiment_name
        )
        try:
            os.makedirs(self.save_path, exist_ok=True)
        except:
            pass

    def log(
        self,
        data: dict,
        step: int,
        batch: DataProto,
        *args,
        **kwargs
    ):
        import json
        if 'tokenizer' not in kwargs:
            raise ValueError("Please provide a tokenizer.")
        
        tokenizer = kwargs['tokenizer']
        with open(os.path.join(self.save_path, f"rollout_data_rank0.jsonl"), "a") as f:       
            for i in range(len(batch)):
                data_item = batch[i]
                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                prompt_str = tokenizer.decode(valid_prompt_ids)
                response_str = tokenizer.decode(valid_response_ids)
                response_tokens = [tokenizer.decode([token]) for token in valid_response_ids]
                cur_sample = {
                    "step": step,
                    "prompt": prompt_str,
                    "response": response_str,
                    "response_tokens": response_tokens,
                    "logprobs": data_item.batch['old_log_probs'][:valid_response_length].cpu().tolist(),
                    "ref_logprobs": data_item.batch['ref_log_prob'][:valid_response_length].cpu().tolist(),
                    #"values": data_item.batch['values'][:valid_response_length].cpu().tolist(),
                    "token_rewards": data_item.batch['token_level_rewards'][:valid_response_length].cpu().tolist(),     # with KL penalty
                    "reward": data_item.batch['token_level_scores'][:valid_response_length].cpu().sum().item(),         # without KL penalty"
                }
                
                if "ground_truth" in data_item.non_tensor_batch['reward_model']:
                    cur_sample["ground_truth"] = data_item.non_tensor_batch['reward_model']["ground_truth"]
                
                f.write(f"{json.dumps(cur_sample, ensure_ascii=False)}\n")

    def finish(self):
        self.writer.close()

class _RLLoggingBoardBackend:
    def __init__(self, script_path="/logger/rl_logging_board.py", port=8081, headless=True, quiet=True):
        """
        初始化 RLLoggingBoard 后台控制类

        :param script_path: RLLoggingBoard 的脚本路径
        :param port: Streamlit 运行端口
        :param headless: 是否为无头模式（不打开浏览器）
        :param quiet: 是否静默运行（不输出 Streamlit 日志）
        """
        self.script_path = script_path
        self.port = port
        self.headless = headless
        self.quiet = quiet
        self.proc = None

    def _set_streamlit_headless(self):
        """设置 ~/.streamlit/config.toml 含主题样式和服务配置"""
        config_dir = os.path.expanduser("~/.streamlit")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "config.toml")
        with open(config_path, "w") as f:
            f.write(
                f"""
[server]
headless = {str(self.headless).lower()}
browser.gatherUsageStats = false
websocket_ping_timeout = 300
port = {self.port}

[theme]
base = "dark"
backgroundColor = "#171719"
secondaryBackgroundColor = "#202025"
primaryColor = "#AAAAAD"
font = "serif"
textColor = "#ceced2"
                """.strip()
            )

    def start(self):
        """启动 RLLoggingBoard"""
        self._set_streamlit_headless()
        cmd = ["streamlit", "run", self.script_path, "--server.port", str(self.port)]

        stdout = subprocess.DEVNULL if self.quiet else None

        self.proc = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )
        time.sleep(2)
        print(f"[RLLoggingBoard] 已启动，监听端口 {self.port}")

    def stop(self):
        """关闭 RLLoggingBoard"""
        if self.proc:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            print("[RLLoggingBoard] 已关闭")
            self.proc = None
