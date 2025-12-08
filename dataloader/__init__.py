from llama_datasets import dataset_factory

from .lru import *
from .llm import *
from .utils import *

DATALOADERS = {
    'lru': LRUDataloader,
    'llm': LLMDataloader,
}


def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.model_code]
    dataloader = dataloader(args, dataset)
    if args.model_code == 'llm':
        train, val, test, test_retrieval = dataloader.get_pytorch_dataloaders()
        tokenizer = dataloader.tokenizer
    else:
        train, val, test = dataloader.get_pytorch_dataloaders()
        test_retrieval = None
        tokenizer = None
    return train, val, test, tokenizer, test_retrieval


def test_subset_dataloader_loader(args):
    dataset = dataset_factory(args)
    if args.model_code == 'lru':
        dataloader = LRUDataloader(args, dataset)
    elif args.model_code == 'llm':
        dataloader = LLMDataloader(args, dataset)

    return dataloader.get_pytorch_test_subset_dataloader()
3