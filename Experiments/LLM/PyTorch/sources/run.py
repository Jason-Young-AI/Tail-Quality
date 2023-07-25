import json
import time
import argparse

import pandas as pd
import tensor_parallel as tp
import torch

from tqdm import tqdm
from pathlib import Path
from transformers import LlamaForCausalLM, LlamaTokenizer


def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    input_tokens = {k:input_tokens[k] for k in input_tokens if k in ["input_ids", "attention_mask"]}
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens

def batch_split(prompts, batch_size):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_size:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts


def batch_infer(llm, tokenizer, prompts, batch_size=1):
    answers = []
    times = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        start_time = time.perf_counter()
        encode_inputs = prepare_input(tokenizer, batch_input)
        stop_time = time.perf_counter()
        pre_time = stop_time - start_time
        start_time = time.perf_counter()
        outputs = llm.generate(**encode_inputs, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)
        stop_time = time.perf_counter()
        inference_time = stop_time - start_time
        start_time = time.perf_counter()
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        stop_time = time.perf_counter()
        post_time = stop_time - start_time
        #print(inference_time)
        times.extend(
            [
                dict(
                    preprocess_time=pre_time,
                    postprocess_time=post_time,
                    inference_time=inference_time,
                )
                for i in range(len(batch_input))
            ]
        )
    answers = [answer[-1] for answer in answers]
    assert len(answers) == len(times)
    return answers, times


def format_example(data_frame, index, include_answer=True):
    choices = ["A", "B", "C", "D"]
    assert len(choices) == data_frame.shape[1] - 2
    prompt = data_frame.iloc[index, 0]
    for i in range(len(choices)):
        prompt += f"\n{choices[i]}. {data_frame.iloc[index, i+1]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {data_frame.iloc[index, len(choices)+1]}\n\n"
    return prompt


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def gen_prompt(train_data_frame, subject, number_train=-1):
    prompt = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    if number_train == -1:
        number_train = train_data_frame.shape[0]
    for i in range(number_train):
        prompt += format_example(train_data_frame, i)
    return prompt


def load_dataset(dev_path, subset_path, mode='val'):
    tasks = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science',
        'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering',
        'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
        'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics',
        'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history',
        'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics',
        'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
        'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
    ]
    data_paths = list()
    for task in tasks:
        dev_task_path = dev_path.joinpath(f'{task}_dev.csv')
        assert dev_task_path.is_file(), f"MMLU dev task path {dev_task_path} does not exist"

        subset_task_path = subset_path.joinpath(f'{task}_{mode}.csv')
        assert subset_task_path.is_file(), f"MMLU {mode} task path {subset_task_path} does not exist"
        data_paths.append((dev_task_path, subset_task_path))

    return tasks, data_paths


def load_llm(model_path):
    n_gpus = torch.cuda.device_count()

    # we use tensor parallel for loading llama
    tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False, padding_side="left")
    
    model = LlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage = True, torch_dtype=torch.float16)
    model = tp.tensor_parallel(model, [i for i in range(n_gpus)])

    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1

    model.eval()

    return model, tokenizer


def run_all(start_i, run_number, batch_size, number_train, tasks, data_paths, results_basepath, llm, tokenizer, write_main_indices=[1,], warm_run=1):
    assert isinstance(write_main_indices, set), "Wrong type of argument \"write_main_indices\": \"{write_main_indices}\""
    assert len(tasks) == len(data_paths), "Fatal Error!"
    print(f" + Total warmup round - {warm_run}")
    print(f" + After that process will start from round {start_i} to {run_number}")
    for true_i, i in enumerate(range(start_i, run_number + warm_run + 1)):
        main_results = dict()
        time_results = dict()
        print(f"{f' . WarmRun[{true_i+1}/{warm_run}]' if true_i < warm_run else f' - Response {i - warm_run}/{run_number}'}")
        for task, (dev_path, subset_path) in zip(tasks, data_paths):
            print(f' - Testing {task} ...')
            records = []
            dev_data_frame = pd.read_csv(dev_path, header=None)[:number_train]
            subset_data_frame = pd.read_csv(subset_path, header=None)
            for row_i in range(subset_data_frame.shape[0]):
                # get prompt and make sure it fits
                prompt_end = format_example(subset_data_frame, row_i, include_answer=False)
                train_prompt = gen_prompt(dev_data_frame, task, number_train)
                prompt = train_prompt + prompt_end
                prompt_token_len = len(tokenizer.tokenize(prompt)) + 1 # bos token
                prompt_qests_num = len(prompt.split("\n\n")) - 1 # without begining statement
                while prompt_token_len > 2048:
                    prompt_split = prompt.split("\n\n")
                    prompt_split.pop(1)
                    prompt = '\n\n'.join(prompt_split)
                    prompt_token_len = len(tokenizer.tokenize(prompt)) + 1
                    prompt_qests_num = len(prompt.split("\n\n")) - 1
                label = subset_data_frame.iloc[row_i, subset_data_frame.shape[1]-1]
                records.append({'prompt': prompt, 'answer': label, 'token_length': prompt_token_len, 'question_number': prompt_qests_num})

            pred_answers, pred_times = batch_infer(llm, tokenizer, [record['prompt'] for record in records], batch_size)
            gold_answers = [record['answer'] for record in records]
            token_lengths = [record['token_length'] for record in records]
            question_numbers = [record['question_number'] for record in records]
            main_results[task] = {'pred_answers': pred_answers, 'gold_answers': gold_answers, 'token_lengths': token_lengths, 'question_numbers': question_numbers}
            time_results[task] = {'pred_times': pred_times}

        if true_i < warm_run:
            pass
        else:
            write_i = i - warm_run
            if write_i in write_main_indices:
                main_path = results_basepath.with_name(results_basepath.name + f".main.{write_i}")
            time_path = results_basepath.with_name(results_basepath.name + f".time.{write_i}")

            if write_i in write_main_indices:
                print(f"Write No.{write_i} MAIN results in File:\"{main_path}\".")
                main_results_json = json.dumps(main_results, indent=2)
                with open(main_path, 'w') as f:
                    f.write(main_results_json)
                
            print(f"Write No.{write_i} Others results in File:\"{time_path}\".")
            time_results_json = json.dumps(time_results, indent=2)
            with open(time_path, 'w') as f:
                f.write(time_results_json)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multiple times on the Whole Test Dataset")
    parser.add_argument('--run-number', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--results-basepath', type=str, required=True)
    parser.add_argument('--model-path', type=str, default="lmsys/vicuna-13b-v1.3")
    parser.add_argument('--start-i', type=int, default=1)
    parser.add_argument('--write-main-indices', type=int, nargs='+', default=[1,])
    parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('--warm-run', type=int, default=1)
    parser.add_argument('--run-mode', type=str, default="val", choices=["val", "test"])
    parser.add_argument('--number-train', type=int, default=5)
    args = parser.parse_args()

    if args.local:
        if args.model_path == 'lmsys/vicuna-13b-v1.3':
            print(f"Using HuggingFace [Online] Pretrained Model: \'lmsys/vicuna-13b-v1.3\'")
        else:
            print(f"Using HuggingFace [Offline] Pretrained Model: \'lmsys/vicuna-13b-v1.3\'")
            model_path = Path(args.model_path)
            assert model_path.is_file(), f"Model Weights path {model_path.name} does not exist."


    dataset_root = Path(args.dataset_path)
    assert dataset_root.is_dir(), f"provided MMLU path {dataset_root} does not exist"

    results_basepath = Path(args.results_basepath)
    assert results_basepath.parent.is_dir(), f"provided results saving dir {results_basepath.parent.name} does not exist."

    dev_path = dataset_root.joinpath("dev")
    assert dev_path.is_dir(), f"MMLU dev path {dev_path} does not exist"

    if args.run_mode == 'val':
        val_path = dataset_root.joinpath("val")
        assert val_path.is_dir(), f"MMLU val path {val_path} does not exist"
        tasks, data_paths = load_dataset(dev_path, val_path)
    if args.run_mode == 'test':
        test_path = dataset_root.joinpath("test")
        assert test_path.is_dir(), f"MMLU test path {test_path} does not exist"
        tasks, data_paths = load_dataset(dev_path, test_path)

    indices = set()
    for i in args.write_main_indices:
        if args.start_i <= i and i <= args.run_number:
            indices.add(i)

    if args.local:
        llm, tokenizer = load_llm(args.model_path)
        run_all(args.start_i, args.run_number, args.batch_size, args.number_train, tasks, data_paths, results_basepath, llm, tokenizer, write_main_indices=indices, warm_run=args.warm_run)
    else:
        print("There is no implementation of Serve Mode!")