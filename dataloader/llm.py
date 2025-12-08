from .base import AbstractDataloader
from .utils import Prompter
from .collate_fns import llama_collate_fn_w_truncation

import torch
import random
import numpy as np
import torch.utils.data as data_utils

import os
import pickle
import transformers
from transformers import AutoTokenizer
from transformers.models.llama.tokenization_llama import DEFAULT_SYSTEM_PROMPT
from trainer import absolute_recall_mrr_ndcg_for_ks


def worker_init_fn(worker_id):
    random.seed(int(np.random.get_state()[1][0]) + worker_id)                                                      
    np.random.seed(int(np.random.get_state()[1][0]) + worker_id)


# the following prompting is based on alpaca
def generate_and_tokenize_eval(args, data_point, tokenizer, prompter):
    in_prompt = prompter.generate_prompt(data_point["system"],
                                         data_point["input"])
    tokenized_full_prompt = tokenizer(in_prompt,
                                      truncation=True,
                                      max_length=args.llm_max_text_len,
                                      padding=False,
                                      return_tensors=None)
    tokenized_full_prompt["labels"] = ord(data_point["output"]) - ord('A')
    
    return tokenized_full_prompt


def generate_and_tokenize_train(args, data_point, tokenizer, prompter):
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt,
                           truncation=True,
                           max_length=args.llm_max_text_len,
                           padding=False,
                           return_tensors=None)
        if (result["input_ids"][-1] != tokenizer.eos_token_id and add_eos_token):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    full_prompt = prompter.generate_prompt(data_point["system"],
                                           data_point["input"],
                                           data_point["output"])
    tokenized_full_prompt = tokenize(full_prompt, add_eos_token=True)
    if not args.llm_train_on_inputs:
        tokenized_full_prompt["labels"][:-2] = [-100] * len(tokenized_full_prompt["labels"][:-2])
    
    return tokenized_full_prompt


def seq_to_token_ids(args, seq, candidates, label, text_dict, tokenizer, prompter, eval=False):
    def truncate_title(title):
        title_ = tokenizer.tokenize(title)[:args.llm_max_title_len]
        title = tokenizer.convert_tokens_to_string(title_)
        return title

    seq_t = ' \n '.join(['(' + str(idx + 1) + ') ' + truncate_title(text_dict[item]) 
                       for idx, item in enumerate(seq)])
    can_t = ' \n '.join(['(' + chr(ord('A') + idx) + ') ' + truncate_title(text_dict[item])
                       for idx, item in enumerate(candidates)])
    output = chr(ord('A') + candidates.index(label))  # ranking only
    
    data_point = {}
    data_point['system'] = args.llm_system_template if args.llm_system_template is not None else DEFAULT_SYSTEM_PROMPT
    data_point['input'] = args.llm_input_template.format(seq_t, can_t)
    data_point['output'] = output
    
    if eval:
        return generate_and_tokenize_eval(args, data_point, tokenizer, prompter)
    else:
        return generate_and_tokenize_train(args, data_point, tokenizer, prompter)


class LLMDataloader():
    def __init__(self, args, dataset):
        self.args = args
        self.rng = np.random
        self.save_folder = dataset._get_preprocessed_folder_path()
        seq_dataset = dataset.load_dataset()
        self.train = seq_dataset['train']
        self.val = seq_dataset['val']
        self.test = seq_dataset['test']
        self.umap = seq_dataset['umap']
        self.smap = seq_dataset['smap']
        self.text_dict = seq_dataset['meta']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)
        
        args.num_items = self.item_count
        self.max_len = args.llm_max_history
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llm_base_tokenizer, cache_dir=args.llm_cache_dir)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.clean_up_tokenization_spaces = True
        self.prompter = Prompter()
        
        self.llm_retrieved_path = args.llm_retrieved_path
        print('Loading retrieved file from {}'.format(self.llm_retrieved_path))
        retrieved_file_path = os.path.join(args.llm_retrieved_path, 'retrieved.pkl')
        if not os.path.exists(retrieved_file_path):
            raise FileNotFoundError(f"Retrieved file not found: {retrieved_file_path}")
        
        retrieved_file = pickle.load(open(retrieved_file_path, 'rb'))
        
        required_keys = ['val_probs', 'val_labels', 'val_metrics', 'test_probs', 'test_labels', 'test_metrics']
        missing_keys = [key for key in required_keys if key not in retrieved_file]
        if missing_keys:
            raise KeyError(f"Missing keys in retrieved file: {missing_keys}")
        
        print('******************** Constructing Validation Subset ********************')
        self.val_probs = retrieved_file['val_probs']
        self.val_labels = retrieved_file['val_labels']
        self.val_metrics = retrieved_file['val_metrics']

        # Convert to tensors once instead of repeatedly
        val_probs_tensor = torch.tensor(self.val_probs)
        val_labels_tensor = torch.tensor(self.val_labels).view(-1)

        # Find top-k indices for all users at once using vectorized operations
        k = self.args.llm_negative_sample_size + 1
        topk_indices = torch.topk(val_probs_tensor, k, dim=1).indices

        # Use vectorized comparison instead of nested loops
        valid_mask = (val_labels_tensor.unsqueeze(1) == topk_indices).any(dim=1)
        self.val_users = [u for u, valid in enumerate(valid_mask.tolist(), start=1) if valid]

        # Pre-compute candidates for all users in batch
        all_candidates = topk_indices.tolist()
        self.val_candidates = [all_candidates[u-1] for u in self.val_users]

        # Debug: limit validation users if specified
        if self.args.debug_max_val_users is not None and len(self.val_users) > self.args.debug_max_val_users:
            self.val_users = self.val_users[:self.args.debug_max_val_users]
            self.val_candidates = self.val_candidates[:self.args.debug_max_val_users]
            print(f'DEBUG MODE: Limited to {self.args.debug_max_val_users} validation users')

        print(f'Validation users: {len(self.val_users)} out of {len(self.val_probs)}')
        if len(self.val_users) == 0:
            print('WARNING: No valid validation users found. Check val_probs and val_labels.')

        print('******************** Constructing Test Subset ********************')
        self.test_probs = retrieved_file['test_probs']
        self.test_labels = retrieved_file['test_labels']
        self.test_metrics = retrieved_file['test_metrics']
        
        # Convert to tensors once instead of repeatedly
        test_probs_tensor = torch.tensor(self.test_probs)
        test_labels_tensor = torch.tensor(self.test_labels).view(-1)

        # Find top-k indices for all users at once using vectorized operations
        k = self.args.llm_negative_sample_size + 1
        test_topk_indices = torch.topk(test_probs_tensor, k, dim=1).indices

        # Use vectorized comparison instead of nested loops
        test_valid_mask = (test_labels_tensor.unsqueeze(1) == test_topk_indices).any(dim=1)
        self.test_users = [u for u, valid in enumerate(test_valid_mask.tolist(), start=1) if valid]

        # Pre-compute candidates for all users in batch
        all_test_candidates = test_topk_indices.tolist()
        self.test_candidates = [all_test_candidates[u-1] for u in self.test_users]
        
        # Debug: limit test users if specified
        if self.args.debug_max_test_users is not None and len(self.test_users) > self.args.debug_max_test_users:
            self.test_users = self.test_users[:self.args.debug_max_test_users]
            self.test_candidates = self.test_candidates[:self.args.debug_max_test_users]
            print(f'DEBUG MODE: Limited to {self.args.debug_max_test_users} test users')
        
        self.non_test_users = [u for u, (p, l) in enumerate(zip(self.test_probs, self.test_labels), start=1) \
                               if l not in torch.topk(torch.tensor(p), self.args.llm_negative_sample_size+1).indices]
        self.test_retrieval = {
            'original_size': len(self.test_probs),
            'retrieval_size': len(self.test_candidates),
            'original_metrics': self.test_metrics,
            'retrieval_metrics': absolute_recall_mrr_ndcg_for_ks(
                torch.tensor(self.test_probs)[torch.tensor(self.test_users)-1],
                torch.tensor(self.test_labels)[torch.tensor(self.test_users)-1],
                self.args.metric_ks,
            ),
            'non_retrieval_metrics': absolute_recall_mrr_ndcg_for_ks(
                torch.tensor(self.test_probs)[torch.tensor(self.non_test_users)-1],
                torch.tensor(self.test_labels)[torch.tensor(self.non_test_users)-1],
                self.args.metric_ks,
            ),
        }
        print(f'Test users: {len(self.test_users)} out of {len(self.test_probs)}')

        # Remove these lines as loaders don't exist yet and trainer will set collate_fn

    @classmethod
    def code(cls):
        return 'llm'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader, self.test_retrieval

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.lora_micro_batch_size,
                                           shuffle=True, pin_memory=True, num_workers=self.args.num_workers,
                                           worker_init_fn=worker_init_fn)
        return dataloader

    def _get_train_dataset(self):
        dataset = LLMTrainDataset(self.args, self.train, self.max_len, self.rng,
                                  self.text_dict, self.tokenizer, self.prompter)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                           pin_memory=True, num_workers=self.args.num_workers)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = LLMValidDataset(self.args, self.train, self.val, self.max_len, self.rng, \
                                      self.text_dict, self.tokenizer, self.prompter, self.val_users, \
                                      self.val_candidates)
        elif mode == 'test':
            dataset = LLMTestDataset(self.args, self.train, self.val, self.test, self.max_len, \
                                     self.rng, self.text_dict, self.tokenizer, self.prompter, self.test_users, \
                                     self.test_candidates)
        return dataset


class LLMTrainDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, max_len, rng, text_dict, tokenizer, prompter):
        self.args = args
        self.max_len = max_len
        self.num_items = args.num_items
        self.rng = rng
        self.text_dict = text_dict
        self.tokenizer = tokenizer
        self.prompter = prompter

        self.all_seqs = []
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            for i in range(2, len(seq)+1):
                self.all_seqs += [seq[:i]]

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        tokens = self.all_seqs[index]
        answer = tokens[-1]
        original_seq = tokens[:-1]
        
        seq = original_seq[-self.max_len:]
        cur_idx, candidates = 0, [answer]
        samples = self.rng.randint(1, self.args.num_items+1, size=5*self.args.llm_negative_sample_size)
        while len(candidates) < self.args.llm_negative_sample_size + 1:
            item = samples[cur_idx]
            cur_idx += 1
            if item in original_seq or item == answer: continue
            else: candidates.append(item)
        self.rng.shuffle(candidates)

        return seq_to_token_ids(self.args, seq, candidates, answer, self.text_dict, \
                                self.tokenizer, self.prompter, eval=False)


class LLMValidDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, u2answer, max_len, rng, text_dict, tokenizer, prompter, val_users, val_candidates):
        self.args = args
        self.u2seq = u2seq
        self.u2answer = u2answer
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.rng = rng
        self.text_dict = text_dict
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.val_users = val_users
        self.val_candidates = val_candidates

    def __len__(self):
        return len(self.val_users)

    def __getitem__(self, index):
        user = self.val_users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user][0]
        
        seq = seq[-self.max_len:]
        candidates = self.val_candidates[index]
        assert answer in candidates
        # self.rng.shuffle(candidates)
        
        return seq_to_token_ids(self.args, seq, candidates, answer, self.text_dict, self.tokenizer, self.prompter, eval=True)


class LLMTestDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, u2val, u2answer, max_len, rng, text_dict, tokenizer, prompter, test_users, test_candidates):
        self.args = args
        self.u2seq = u2seq
        self.u2val = u2val
        self.u2answer = u2answer
        self.users = sorted(u2seq.keys())
        self.max_len = max_len
        self.rng = rng
        self.text_dict = text_dict
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.test_users = test_users
        self.test_candidates = test_candidates
    
    def __len__(self):
        return len(self.test_users)
    
    def __getitem__(self, index):
        user = self.test_users[index]
        seq = self.u2seq[user] + self.u2val[user]
        answer = self.u2answer[user][0]

        seq = seq[-self.max_len:]
        candidates = self.test_candidates[index]
        assert answer in candidates
        # self.rng.shuffle(candidates)

        return seq_to_token_ids(self.args, seq, candidates, answer, self.text_dict, self.tokenizer, self.prompter, eval=True)