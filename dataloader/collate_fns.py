import torch


def llama_collate_fn_w_truncation(llm_max_length, eval=False):
    def llama_collate_fn(batch):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        example_max_length = max([len(batch[idx]['input_ids']) for idx in range(len(batch))])
        max_length = min(llm_max_length, example_max_length)
        
        for i in range(len(batch)):
            input_ids = batch[i]['input_ids']
            attention_mask = batch[i]['attention_mask']
            labels = batch[i]['labels']
            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
                attention_mask = attention_mask[-max_length:]
                if not eval: labels = labels[-max_length:]
            elif len(input_ids) < max_length:
                padding_length = max_length - len(input_ids)
                input_ids = [0] * padding_length + input_ids
                attention_mask = [0] * padding_length + attention_mask
                if not eval: labels = [-100] * padding_length + labels

            all_input_ids.append(torch.tensor(input_ids).long())
            all_attention_mask.append(torch.tensor(attention_mask).long())
            all_labels.append(torch.tensor(labels).long())
        
        return {
            'input_ids': torch.vstack(all_input_ids),
            'attention_mask': torch.vstack(all_attention_mask),
            'labels': torch.vstack(all_labels)
        }
    return llama_collate_fn
