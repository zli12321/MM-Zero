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

import importlib.util
import inspect
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Callable, List, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


# ---------------------------------------------------------------------------
# Helper: extract extra fields from non_tensor_batch
# ---------------------------------------------------------------------------
_EXTRA_KEYS = ("question", "description_answers", "images", "image", "path", "raw_prompt_text")


def _get_extra_arrays(data: DataProto) -> dict:
    """Return a dict of {key: array_or_None} for commonly used extra fields."""
    extras = {}
    for key in _EXTRA_KEYS:
        arr = data.non_tensor_batch.get(key, None)
        extras[key] = arr
    # Normalise images: prefer "images" over "image"
    if extras["images"] is None and extras["image"] is not None:
        extras["images"] = extras["image"]
    # Fallback: if "question" is not in the dataset but "raw_prompt_text" is,
    # use the raw prompt text as the question field (useful for codegen reward
    # where the full proposal text is the prompt itself).
    if extras["question"] is None and extras["raw_prompt_text"] is not None:
        extras["question"] = extras["raw_prompt_text"]
    return extras


# ---------------------------------------------------------------------------
# Sequential mixin — new dict-based interface (enhanced with extra fields)
# ---------------------------------------------------------------------------
class SequentialFunctionRewardManagerMixin:
    reward_fn: SequentialRewardFunction

    def compute_reward_sequential(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)

        extras = _get_extra_arrays(data)

        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            reward_input = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            # Attach extra fields when available
            if extras["question"] is not None:
                reward_input["question"] = extras["question"][i]
            if extras["description_answers"] is not None:
                reward_input["description_answers"] = extras["description_answers"][i]
            if extras["images"] is not None:
                reward_input["images"] = extras["images"][i]

            score = self.reward_fn(reward_input)
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


# ---------------------------------------------------------------------------
# Batch mixin — new dict-based interface (enhanced with extra fields)
# ---------------------------------------------------------------------------
class BatchFunctionRewardManagerMixin:
    reward_fn: BatchRewardFunction

    def compute_reward_batch(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)

        extras = _get_extra_arrays(data)

        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            reward_input = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            if extras["question"] is not None:
                reward_input["question"] = extras["question"][i]
            if extras["description_answers"] is not None:
                reward_input["description_answers"] = extras["description_answers"][i]
            if extras["images"] is not None:
                reward_input["images"] = extras["images"][i]

            reward_inputs.append(reward_input)

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


# ---------------------------------------------------------------------------
# Legacy batch mixin — old VisPlay positional-args interface
#   compute_score(predicts, ground_truths, questions, description_answers,
#                 format_weight=0.1, images=None)
# ---------------------------------------------------------------------------
class LegacyBatchFunctionRewardManagerMixin:
    """Supports the old VisPlay-style interface with positional list arguments."""

    reward_fn: Callable

    def compute_reward_legacy_batch(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        response_str_list: List[str] = []
        ground_truth_list: List[str] = []
        questions_list: List[str] = []
        description_answers_list: List[str] = []
        images_list: list = []

        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)

        extras = _get_extra_arrays(data)

        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str_list.append(
                self.tokenizer.decode(valid_response_ids, skip_special_tokens=self.config.skip_special_tokens)
            )
            ground_truth_list.append(data.non_tensor_batch["ground_truth"][i])
            questions_list.append("" if extras["question"] is None else extras["question"][i])
            description_answers_list.append("" if extras["description_answers"] is None else extras["description_answers"][i])
            images_list.append(None if extras["images"] is None else extras["images"][i])

        # Inspect the function signature to decide which args to pass
        target_fn = self.reward_fn.func if isinstance(self.reward_fn, partial) else self.reward_fn
        param_names = list(inspect.signature(target_fn).parameters.keys())

        base_args = [response_str_list, ground_truth_list]
        call_kwargs = {}
        if "questions" in param_names:
            call_kwargs["questions"] = questions_list
        elif "question" in param_names:
            call_kwargs["question"] = questions_list
        if "description_answers" in param_names:
            call_kwargs["description_answers"] = description_answers_list
        if "images" in param_names:
            call_kwargs["images"] = images_list
        elif "image" in param_names:
            call_kwargs["image"] = images_list

        try:
            scores = self.reward_fn(*base_args, **call_kwargs)
        except TypeError:
            # Fallback attempts: positional expansion
            try:
                scores = self.reward_fn(response_str_list, ground_truth_list, questions_list)
            except TypeError:
                try:
                    scores = self.reward_fn(
                        response_str_list, ground_truth_list, questions_list, description_answers_list
                    )
                except TypeError:
                    try:
                        scores = self.reward_fn(
                            response_str_list, ground_truth_list, questions_list,
                            description_answers_list, images_list,
                        )
                    except TypeError:
                        # Last resort: original 2-arg (legacy)
                        scores = self.reward_fn(response_str_list, ground_truth_list)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


# ---------------------------------------------------------------------------
# Auto reward manager — routes to the right mixin based on REWARD_TYPE
# ---------------------------------------------------------------------------
class AutoRewardManager(
    LegacyBatchFunctionRewardManagerMixin,
    BatchFunctionRewardManagerMixin,
    SequentialFunctionRewardManagerMixin,
):
    """Reward manager for rule-based reward.

    Supported REWARD_TYPE values (set as module-level constant in the reward file):
      - "batch"          : new dict-based batch interface  (default)
      - "sequential"     : new dict-based sequential interface
      - "legacy_batch"   : old VisPlay positional-args batch interface

    If REWARD_TYPE is missing, the manager inspects the function signature:
      - First param named "reward_inputs" → "batch"
      - Otherwise                         → "legacy_batch"
    """

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        reward_name = getattr(module, "REWARD_NAME", "unknown")

        # --- Determine reward type ---
        reward_type = getattr(module, "REWARD_TYPE", None)
        if reward_type is None:
            # Auto-detect from function signature
            first_param = list(inspect.signature(reward_fn).parameters.keys())[0]
            if first_param == "reward_inputs":
                reward_type = "batch"
            else:
                reward_type = "legacy_batch"
            print(f"REWARD_TYPE not set in module; auto-detected as '{reward_type}' "
                  f"(first param name: '{first_param}').")

        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        print(f"Reward name: {reward_name}, reward type: {reward_type}.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.reward_type = reward_type
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        if self.reward_type == "batch":
            return self.compute_reward_batch(data)
        elif self.reward_type == "sequential":
            return self.compute_reward_sequential(data)
        elif self.reward_type == "legacy_batch":
            return self.compute_reward_legacy_batch(data)
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}.")
