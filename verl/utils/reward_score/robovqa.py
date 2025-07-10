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
import re

# from mathruler.grader import extract_boxed_content, grade_answer



def single_choice_reward_fn(to_be_evaluated: str, reference: str) -> float:
    """Check if the answer is correct using single choice logic."""
    reward = 0.0
    try:
        # Extract answer from solution if it has think/answer tags
        sol_match = re.search(r"<answer>(.*?)</answer>", reference, re.DOTALL)
        ground_truth_answer = sol_match.group(1).strip() if sol_match else reference.strip()

        # Extract answer from content if it has think/answer tags
        content_match = re.search(r"<answer>(.*?)</answer>", to_be_evaluated, re.DOTALL)
        student_answer = (
            content_match.group(1).strip() if content_match else to_be_evaluated.strip()
        )

        # Compare the extracted answers (case insensitive)
        if student_answer.lower() == ground_truth_answer.lower():
            reward = 1.0
    except Exception:
        reward = 0.0

    return reward

def format_reward_fn(to_be_evaluated):
    try:
        pattern = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(pattern, to_be_evaluated, re.DOTALL)
        # if the format is not correct, reward is 0
        if match is None or len(match.groups()) != 2:
            return 0.0
        else:
            return 1.0
    except Exception as e:  # noqa: BLE001
        print("Exception in format_reward_func: %s", e) 
        return 0.0





def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
    """
    Compute reward score using cosmos-rl style format and single choice reward functions.
    
    Args:
        predict_str: The model's prediction string
        ground_truth: The ground truth answer
        use_boxed: Whether to use boxed format (kept for compatibility, but not used)
        format_score: Weight for format reward component
        
    Returns:
        Combined reward score (format_reward * format_score + correctness_reward * (1 - format_score))
    """
    

    
    # Compute format reward
    format_reward = format_reward_fn(predict_str)
    
    # Compute correctness reward
    correctness_reward = single_choice_reward_fn(predict_str, ground_truth)
    
    # Combine rewards: format check gets format_score weight, correctness gets the rest
    # If format is wrong, still give some reward for correct content
    total_reward = format_reward * format_score + correctness_reward * (1 - format_score)
    
    return total_reward