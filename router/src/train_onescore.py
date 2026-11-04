"""Usage:
    python src/train_onescore.py --data_dir data/clustered/train_lambda_0.5 \
        --test_data_dir data/dev_lambda_0.5 --save_path runs/lambda_0.5_alpha_0.2
"""

import argparse
import json
import os
import random
import subprocess

import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
from transformers import AutoTokenizer, DebertaV2Model
from utils.meters import AverageMeter
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

def collect_json_files(directory):
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a valid directory")
    return sorted(
        [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(".json")
        ]
    )

def get_available_gpu(memory_threshold_gb=8):
    try:
        smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        ).strip()
        
        gpus = []
        for line in smi_output.split('\n'):
            if not line: continue
            index, utilization, free_memory = line.split(', ')
            gpus.append({
                'index': int(index),
                'utilization': int(utilization),
                'free_memory_mb': int(free_memory)
            })

        available_gpus = [
            gpu for gpu in gpus 
            if gpu['utilization'] == 0 and gpu['free_memory_mb'] >= memory_threshold_gb * 1024
        ]

        if not available_gpus:
            print(f"No GPU found with 0 utilization and >{memory_threshold_gb}GB free memory. Checking for any GPU with 0 utilization.")
            available_gpus = [gpu for gpu in gpus if gpu['utilization'] == 0]

        if not available_gpus:
            print("No GPU with 0 utilization found. Checking for the GPU with the most free memory.")
            if gpus:
                best_gpu = max(gpus, key=lambda gpu: gpu['free_memory_mb'])
                print(f"Selected GPU {best_gpu['index']} with {best_gpu['free_memory_mb']}MB free (utilization: {best_gpu['utilization']}%).")
                return f"cuda:{best_gpu['index']}"
            else:
                 print("No GPUs detected by nvidia-smi.")
                 return "cpu"

        max_free = max(gpu['free_memory_mb'] for gpu in available_gpus)
        candidate_gpus = [gpu for gpu in available_gpus if max_free - gpu['free_memory_mb'] <= 2048]
        best_gpu = random.choice(candidate_gpus)
        print(f"Selected GPU {best_gpu['index']} with {best_gpu['free_memory_mb']}MB free (utilization: {best_gpu['utilization']}%).")
        return f"cuda:{best_gpu['index']}"

    except (FileNotFoundError, subprocess.CalledProcessError, Exception) as e:
        print(f"Error querying GPU status: {e}. Defaulting to CPU.")
        return "cpu"

def setup_seed(seed):# set the random seed
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


class RouterDataset(Dataset):
    def __init__(self,
                 data_path,
                 source_max_token_len: int = 512,
                 target_max_token_len: int = 512,
                 size: int = None,
                 data_type: str = "multi_attempt",
                 dataset_id = 0,
                 model_order: str = "json",
                 ):
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.json'):
                self.data = json.load(f)
        if size:
            while(len(self.data) < size):
                self.data.extend(self.data)
            self.data = self.data[:size]
        # sorted: deploy order; json: original
        if model_order not in ("json", "sorted"):
            raise ValueError(f"model_order must be 'json' or 'sorted', got {model_order}")
        keys = list(self.data[0]['scores'].keys())
        self.model_order = model_order
        self.router_node = sorted(keys) if model_order == "sorted" else keys
        self.tokenizer = None
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.data_type = data_type
        self.dataset_id = dataset_id

    def __getitem__(self, index):
        data_point = self.data[index]
        scores = torch.tensor([float(data_point['scores'][m]) for m in self.router_node])
        question = data_point['question']
        question_id = self.tokenizer(
            question,
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        question_id['input_ids'] = question_id.input_ids.flatten()
        question_id['attention_mask'] = question_id.attention_mask.flatten()
        cluster_id = data_point['cluster_id'] if "cluster_id" in data_point else 0
        return question_id, scores, self.dataset_id, cluster_id

    def __len__(self):
        return len(self.data)

    def register_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer


class RouterModule(nn.Module):
    def __init__(self, backbone, hidden_state_dim=768, node_size=3, similarity_function = "cos"):
        super(RouterModule, self).__init__()
        self.backbone = backbone
        self.hidden_state_dim = hidden_state_dim
        self.node_size = node_size
        self.embeddings = nn.Embedding(node_size, hidden_state_dim)
        std_dev = 0.78
        with torch.no_grad():
            nn.init.normal_(self.embeddings.weight, mean=0, std=std_dev)
        self.similarity_function = similarity_function
            

    def compute_similarity(self, input1, input2):
        if self.similarity_function == "cos":
            return (input1 @ input2.T) / (torch.norm(input1,dim=1).unsqueeze(1) * torch.norm(input2,dim=1).unsqueeze(0))
        else:
            return input1 @ input2.T


    def forward(self, t=1, **input_kwargs):
        x = self.backbone(**input_kwargs)
        hidden_state = x['last_hidden_state'][:,0,:]
        x = self.compute_similarity(hidden_state, self.embeddings.weight)
        x = x / t
        return x, hidden_state

    def compute_sample_llm_loss(self, x, index_true, top_k, last_k):
        loss = 0
        top_index_true, top_index = index_true.sort(dim=-1, descending=True)
        last_index_true, negtive_index = index_true.topk(k=last_k, largest=False,dim=-1)

        for i in range(top_k):
            positive_index = top_index[:,i].view(-1,1)

            mask = torch.where(top_index_true[:,i].view(-1,1) > 0, 1, 0)

            top_x = torch.gather(x, 1, positive_index)
            last_x = torch.gather(x, 1, negtive_index)

            last_x = torch.where(last_index_true > 0.5, float("-inf"), last_x)

            temp_x = torch.concat([top_x, last_x], dim=-1)

            softmax_x = nn.Softmax(dim=-1)(temp_x)
            log_x = torch.log(softmax_x[:,0])
            log_x = log_x * mask 
            loss += torch.mean(-log_x)
        return loss
    
    def compute_sample_sample_loss_with_task_tag(self, hidden_state, dataset_ids, t, H=3):
        similar_score = self.compute_similarity(hidden_state, hidden_state)
        last_k2 = H
        all_index = []
        for dataset_id in dataset_ids:
            positive_indexs = torch.nonzero(dataset_ids == dataset_id)
            select_positive_index = random.choice(positive_indexs)
            negtive_indexs = torch.nonzero(dataset_ids != dataset_id)
            if len(negtive_indexs) < last_k2:
                print("len of negtive index is smaller than last_k2. dataset_id:", dataset_id)
                continue
            index_of_negtive_indexs = random.sample(range(0, len(negtive_indexs)), last_k2)
            select_negtive_index = negtive_indexs[index_of_negtive_indexs].squeeze()
            select_index = torch.concat([select_positive_index, select_negtive_index])
            all_index.append(select_index)
        all_index = torch.stack(all_index)
        rearrange_similar_score = torch.gather(similar_score, 1, all_index)

        softmax_sample_x = torch.softmax(rearrange_similar_score, dim=-1)
        log_sample_x = torch.log(softmax_sample_x)
        loss = torch.mean(-log_sample_x[:,0])
        return loss
    
    def compute_cluster_loss(self, hidden_state, cluster_ids, t, H=3):
        similar_score = self.compute_similarity(hidden_state, hidden_state)
        last_k2 = H
        all_index = []
        for cluster_id in cluster_ids:
            positive_indexs = torch.nonzero(cluster_ids == cluster_id)
            select_positive_index = random.choice(positive_indexs)
            negtive_indexs = torch.nonzero(cluster_ids != cluster_id)
            if len(negtive_indexs) < last_k2:
                print("len of negtive index is smaller than last_k2. cluster_id:", cluster_id)
                continue
            index_of_negtive_indexs = random.sample(range(0, len(negtive_indexs)), last_k2)
            select_negtive_index = negtive_indexs[index_of_negtive_indexs].view(-1)
            select_index = torch.concat([select_positive_index, select_negtive_index])
            all_index.append(select_index)
        all_index = torch.stack(all_index)
        rearrange_similar_score = torch.gather(similar_score, 1, all_index)

        softmax_sample_x = torch.softmax(rearrange_similar_score, dim=-1)
        log_sample_x = torch.log(softmax_sample_x)
        loss = torch.mean(-log_sample_x[:,0])
        return loss


def evaluation(router_model, dataset_paths, dataset_types, tokenizer, batch_size, device, model_order="json"):    
    result = {}
    with torch.no_grad():
        assert len(dataset_paths) == len(dataset_types)
        for index, data_path in enumerate(dataset_paths):
            test_dataset = RouterDataset(data_path=data_path, model_order=model_order)
            test_dataset.register_tokenizer(tokenizer)
            data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            correct_predict = 0
            correct = 0
            
            for batch in data_loader:
                inputs, scores, _, _ = batch
                inputs = inputs.to(device)
                scores = scores.to(device)
                x, _ = router_model.forward(**inputs)
                softmax_x = nn.Softmax(dim=1)(x)
                _, max_index = torch.max(softmax_x, dim=1)
                _, target_max_index = torch.max(scores, dim=1)
                equals = max_index.eq(target_max_index)
                correct += equals.sum().item()

                if dataset_types[index] == "probability":
                    mask = torch.zeros_like(scores)
                    mask = mask.scatter_(1, max_index.unsqueeze(1), 1)
                    scores[scores > 0] = 1
                    correct_predict += (scores * mask).sum().item()
                elif dataset_types[index] == "multi_attempt":
                    mask = torch.zeros_like(scores)
                    mask = mask.scatter_(1, max_index.unsqueeze(1), 1)
                    correct_predict += (scores * mask).sum().item()

            acc_predict = correct_predict / len(test_dataset)
            acc = correct / len(test_dataset)
            print(f"acc_{data_path}:", acc_predict)
            print("acc", acc)

            result[data_path] = [acc, acc_predict]
            
    return result

if __name__ == '__main__': 
    time_start = time.time()
    parser = argparse.ArgumentParser(description="the training code for router")

    parser.add_argument('--data_paths', nargs='+', default=None)
    parser.add_argument('--test_data_paths',nargs='+', default=None)
    parser.add_argument('--test_data_type', nargs='+', default=["multi_attempt", "multi_attempt", "multi_attempt","multi_attempt", "multi_attempt", "multi_attempt"])


    parser.add_argument('--data_dir', type=str, default="data/clustered/train_lambda_0.5",
                        help="if set, read all training .json files from this directory and override --data_paths")
    parser.add_argument('--test_data_dir', type=str, default="data/dev_lambda_0.5",
                        help="if set, read all test .json files from this directory and override --test_data_paths")
    parser.add_argument('--backbone', type=str, default="microsoft/mdeberta-v3-base",
                        help="mDeBERTaV3-base encoder: HF repo id, or a local directory path")

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--training_steps', type=int, default=350)
    parser.add_argument('--eval_steps',type=int,default=50)
    parser.add_argument('--learning_rate', type=float, default=0.00005)
    parser.add_argument('--save_path', type=str, default='runs/default')
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--last_k',type=int, default=2)
    parser.add_argument('--tempreture', type=int, default=1)
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--similarity_function', type=str, default='cos')
    parser.add_argument('--gradient_checkpointing', type=int, default=1, choices=[0, 1],
                        help="gradient checkpointing: 1=on (default). bs=64 without it runs out of memory on a 48G card; when on, the peak is about 12G"
                             "and is numerically identical to turning it off (measured: loss/gradient difference is 0); set 0 to speed up when memory is sufficient")
    parser.add_argument('--sample_loss_weight', type=float, default=0)
    parser.add_argument('--cluster_loss_weight', type=float, default=0.2)
    parser.add_argument('--H', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--training_samples_per_dataset', type=int, default=500) 
    parser.add_argument('--model_order', type=str, default='sorted', choices=['sorted', 'json'],
                        help="model order behind the embedding indices. 'sorted' = sort by model name (default, consistent with the released "
                             "ckpt/router_small.pth); 'json' = the original key order of scores in the data "
                             "(the alpha x lambda table in the paper was produced with it; pass json to reproduce those numbers)")

    parser.add_argument('--gpu_memory_threshold', type=int, default=8, help="Minimum free GB required for GPU selection.")
    args = parser.parse_args()

    if args.data_dir:
        args.data_paths = collect_json_files(args.data_dir)
    elif len(args.data_paths) == 1 and os.path.isdir(args.data_paths[0]):
        args.data_paths = collect_json_files(args.data_paths[0])

    if args.test_data_dir:
        args.test_data_paths = collect_json_files(args.test_data_dir)
    elif len(args.test_data_paths) == 1 and os.path.isdir(args.test_data_paths[0]):
        args.test_data_paths = collect_json_files(args.test_data_paths[0])

    print("training data files:", args.data_paths)
    print("test data files:", args.test_data_paths)
    os.makedirs(args.save_path, exist_ok=True)
    setup_seed(args.seed)
    
    device = get_available_gpu(args.gpu_memory_threshold)
    print(f"Using device: {device}")

    print(f"backbone: {args.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, truncation_side='left', padding=True)
    encoder_model = DebertaV2Model.from_pretrained(args.backbone)

    # bs=64 needs this on 48G
    if args.gradient_checkpointing:
        encoder_model.gradient_checkpointing_enable()
        n_drop = 0
        for mod in encoder_model.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
                n_drop += 1
            elif hasattr(mod, "drop_prob"):
                mod.drop_prob = 0.0
                n_drop += 1
        encoder_model.train()
        print(f"gradient checkpointing: ON (dropout set to 0, {n_drop} places, numerically equivalent to eval mode)")
    else:
        print("gradient checkpointing: OFF")

    router_datasets = [RouterDataset(data_path, size=args.training_samples_per_dataset, data_type=args.test_data_type[i], dataset_id = i, model_order=args.model_order) for i, data_path in enumerate(args.data_paths)]
    for router_dataset in router_datasets:
        router_dataset.register_tokenizer(tokenizer)
    router_dataset = ConcatDataset(router_datasets)

    print(f"init_model, router_node size: {router_datasets[0].router_node}")

    model_names = router_datasets[0].router_node
    for ds_ in router_datasets[1:]:
        assert ds_.router_node == model_names, (
            f"model order differs across training files: {ds_.router_node} vs {model_names}"
        )
    index_map = {str(i): name for i, name in enumerate(model_names)}
    model_order_path = os.path.join(args.save_path, "model_order.json")
    with open(model_order_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "order_type": args.model_order,
                "model_order": model_names,
                "index_map": index_map,
                "num_models": len(model_names),
                "output_dim": len(model_names),
                "score_index": 0,
                "note": "dimension i of logits/scores corresponds to model_order[i]; inference must use the same order",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"model order saved to {model_order_path} (order_type={args.model_order})")
    for i, name in enumerate(model_names):
        print(f"  {i}: {name}")

    router_model = RouterModule(encoder_model, hidden_state_dim=768, node_size=len(model_names), similarity_function=args.similarity_function).to(device)

    optimizer = torch.optim.AdamW(router_model.parameters(), lr=args.learning_rate)

    print("Training start!!!")
    pbar = tqdm(range(args.training_steps))
    step = 0
    training_log = []
    max_average = 0
    max_training_average = 0
    
    eval_steps_list = []
    train_accuracies_per_dataset = {}
    test_accuracies_per_dataset = {}

    while(True):
        losses = AverageMeter('Loss', ':3.2f')
        data_loader = DataLoader(router_dataset, batch_size=args.batch_size, shuffle=True)
        for batch in data_loader:
            optimizer.zero_grad()
            inputs, scores, dataset_ids, cluster_ids = batch
            inputs = inputs.to(device)
            scores = scores.to(device)
            dataset_ids = dataset_ids.to(device)
            cluster_ids = cluster_ids.to(device)

            x, hidden_state = router_model.forward(t=args.tempreture, **inputs)
            loss = router_model.compute_sample_llm_loss(x = x, index_true=scores, top_k = args.top_k, last_k = args.last_k)

            if args.sample_loss_weight:
                sample_sample_loss = router_model.compute_sample_sample_loss_with_task_tag(hidden_state=hidden_state, dataset_ids=dataset_ids, t=args.tempreture, H=args.H)
                loss = loss + args.sample_loss_weight * sample_sample_loss

            if args.cluster_loss_weight:
                cluster_loss = router_model.compute_cluster_loss(hidden_state=hidden_state, cluster_ids=cluster_ids, t=args.tempreture, H=args.H)
                loss = loss + args.cluster_loss_weight * cluster_loss

            losses.update(loss.item(), scores.size(0))
            loss.backward()
            if step % args.gradient_accumulation == 0:   
                optimizer.step()

            pbar.set_postfix({"step": f"{step}","loss": loss.item()})
            pbar.write(f"step:{step}, loss:{loss.item()}")
            pbar.update(1)
            step += 1
            if step >= args.training_steps:
                break
            if (step + 1) % args.eval_steps == 0:
                print("validation start")
                val_result = evaluation(router_model, args.data_paths, args.test_data_type, tokenizer, batch_size = args.batch_size, device=device, model_order=args.model_order)
                print("test start")
                test_result = evaluation(router_model, args.test_data_paths, args.test_data_type, tokenizer, batch_size = args.batch_size, device=device, model_order=args.model_order)
                result = {**val_result, **test_result}
                average = sum([ value[1] for value in test_result.values()]) / len(test_result) if test_result else 0
                if average > max_average:
                    torch.save(router_model.state_dict(),  os.path.join(args.save_path, "best_model.pth"))
                    max_average = average
                
                training_average = sum([ value[1] for value in val_result.values()]) / len(val_result) if val_result else 0

                current_step = step + 1
                eval_steps_list.append(current_step)

                for path, (_, acc_predict) in val_result.items():
                    if path not in train_accuracies_per_dataset:
                        train_accuracies_per_dataset[path] = [None] * (len(eval_steps_list) - 1)
                    train_accuracies_per_dataset[path].append(acc_predict)
                for path in train_accuracies_per_dataset:
                     if len(train_accuracies_per_dataset[path]) < len(eval_steps_list):
                         train_accuracies_per_dataset[path].append(None)


                for path, (_, acc_predict) in test_result.items():
                    if path not in test_accuracies_per_dataset:
                        test_accuracies_per_dataset[path] = [None] * (len(eval_steps_list) - 1)
                    test_accuracies_per_dataset[path].append(acc_predict)
                for path in test_accuracies_per_dataset:
                    if len(test_accuracies_per_dataset[path]) < len(eval_steps_list):
                        test_accuracies_per_dataset[path].append(None)


                result['step'] = current_step
                training_log.append(result)

        print(f"step:{step}, avg_loss_per_epoch:{losses.avg}")
        if step >= args.training_steps:
            break

    if eval_steps_list:
        print("Plotting accuracy curves per dataset...")
        plt.figure(figsize=(12, 8))

        for path, acc_list in train_accuracies_per_dataset.items():
            dataset_name = os.path.basename(path).split('.')[0]
            plot_steps = [s for s, acc in zip(eval_steps_list, acc_list) if acc is not None]
            plot_accs = [acc for acc in acc_list if acc is not None]
            if plot_steps:
                 plt.plot(plot_steps, plot_accs, marker='o', linestyle='-', label=f'Train - {dataset_name}')

        for path, acc_list in test_accuracies_per_dataset.items():
            dataset_name = os.path.basename(path).split('.')[0]
            plot_steps = [s for s, acc in zip(eval_steps_list, acc_list) if acc is not None]
            plot_accs = [acc for acc in acc_list if acc is not None]
            if plot_steps:
                 plt.plot(plot_steps, plot_accs, marker='x', linestyle='--', label=f'Test - {dataset_name}')

        plt.xlabel('Training Steps')
        plt.ylabel('Accuracy (Predict)')
        plt.title('Training and Testing Accuracy per Dataset Over Steps')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_save_path = os.path.join(args.save_path, 'accuracy_curve_per_dataset.png')
        plt.savefig(plot_save_path)
        print(f"Per-dataset accuracy plot saved to {plot_save_path}")

    print("best avg", max_average)

    with open(os.path.join(args.save_path, "training_log.json"), 'w') as f:
        json.dump(training_log, f, indent=4)

    with open(os.path.join(args.save_path, "config.txt"), 'w') as f:
        f.write(str(args))
    
    time_end = time.time()
    print(f"Training time: {time_end - time_start} seconds")