from data.webnlg.dataset import GraphDataset

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
import unicodedata as ud
import logging

class GraphDataModule(pl.LightningDataModule):
    """
    Contains PyTorch Lighting’s LightningDataModule which organizes 
    the data loading and preparation steps and offers a clear and standardized interface 
    for the data used in PyTorch Lightning systems.
    It offers the functions:
    * prepare_data: Download data to disk.
    * setup: Load data into GraphDataset(PyTorch's Dataset).
    * train_dataloader: Returns DataLoader object with the dataset.
    and additional parameters like batch size and parameters for mini-batch processing.
    * val_dataloader: Same as train_dataloader, just for validation set.
    * test_dataloader: Same as train_dataloader, just for test set.
    """

    def __init__(self,
                 tokenizer_class,
                 tokenizer_name,
                 cache_dir,
                 data_path,
                 dataset,
                 batch_size,
                 num_data_workers,
                 max_nodes,
                 max_edges,
                 edges_as_classes):
        super().__init__()

        # Setup tokenizer and add special tokens
        # 150 as max tokens was set based on what was found in the grapher.Grapher.sample function
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_name, cache_dir=cache_dir,
                                                         model_max_length=512, legacy=True)
        self.tokenizer.add_tokens('__no_node__')
        self.tokenizer.add_tokens('__no_edge__')
        self.tokenizer.add_tokens('__node_sep__')

        self.batch_size = batch_size
        self.num_data_workers = num_data_workers
        self.data_path = data_path
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.edges_as_classes = edges_as_classes
        self.output_path = os.path.join(data_path, 'processed')
        os.makedirs(self.output_path, exist_ok=True)
        self.dataset = dataset


    # Override
    def prepare_data(self):
        """
        Download the dataset specified in `self.dataset`, splits it into train, dev and test, and saves the splits to disk.
        """

        if self.dataset == 'webnlg':
            self._prepareWebNLG()
        else:
            raise NotImplementedError(f'Unknown dataset {self.dataset}. Only WebNLG dataset has been implemented')
    
    def _prepareWebNLG(self):
        """
        Download the webnlg dataset from HuggingFace and save train, dev and test splits to disk.
        """

        splits = ['train', 'dev', 'test']
        for split in splits:

            text_file = os.path.join(self.output_path, f'{split}.text')
            graph_file = os.path.join(self.output_path, f'{split}.graph')
            if os.path.exists(text_file) and os.path.exists(graph_file):
                continue

            logging.info("Loading the webnlg dataset from Huggingface")
            dataset_tuple = load_dataset("web_nlg", name="release_v3.0_en", cache_dir=f"{os.environ['STORAGE_DRIVE']}/data/core/cache/hug", trust_remote_code=True), 
            if split == 'test':
                test_datasets = dataset_tuple[0][split]
                D = [entry for entry in test_datasets if entry['test_category'] == 'semantic-parsing-test-data-with-refs-en']
            else:
                D = dataset_tuple[0][split]

            normalize = lambda text: ud.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

            logging.info(f"Creating {split}.graph and {split}.text files.")
            triples_to_write = []
            text_to_write = []
            edge_classes_to_write = []
            for entry in D:
                triples = entry['modified_triple_sets']['mtriple_set'][0]
                proc_triples = []
                for triple in triples:
                    sub, rel, obj = triple.split(' | ')
                    obj = normalize(obj.strip('\"').replace('_', ' '))
                    sub = normalize(sub.strip('\"').replace('_', ' '))
                    proc_triples.append(f'__subject__ {sub} __predicate__ {rel} __object__ {obj}')
                    if split == 'train':
                        edge_classes_to_write.append(rel)
                proc_lexs = [v for k, v in entry['lex'].items() if k=='text'][0]
                for lex in proc_lexs:
                    text_to_write.append(normalize(lex))
                    triples_to_write.append(' '.join(proc_triples))


            with open(text_file, 'w') as f:
                f.writelines(s + '\n' for s in text_to_write)

            with open(graph_file, 'w') as f:
                f.writelines(s + '\n' for s in triples_to_write)

            if split == 'train':
                edge_classes_file = os.path.join(self.output_path, 'edge.classes')
                with open(edge_classes_file, 'w') as f:
                    f.writelines(s + '\n' for s in list(set(edge_classes_to_write)) + ['__no_edge__'])


    # Override
    def setup(self, stage=None):
        """
        Load training (train.text, train.graph),
        validation (dev.text, dev.graph), 
        or test data (test.text, test.graph) into GraphDataset
        """

        stage_dataset_mapping = {'fit': 'train', 'validate': 'dev', 'test': 'test'}
        dataset = GraphDataset(tokenizer=self.tokenizer,
                               text_data_path=os.path.join(self.output_path, f"{stage_dataset_mapping[stage]}.text"),
                               graph_data_path=os.path.join(self.output_path, f"{stage_dataset_mapping[stage]}.graph"),
                               edge_classes_path=os.path.join(self.output_path, 'edge.classes'),
                               max_nodes=self.max_nodes,
                               max_edges=self.max_edges,
                               edges_as_classes=self.edges_as_classes)

        if stage == 'fit':
            logging.info("Load training data into GraphDataset.")
            self.dataset_train = dataset
        elif stage == 'validate':
            logging.info("Load validation data (dev) into GraphDataset.")
            self.dataset_dev = dataset
        elif stage == 'test':
            logging.info("Testing the model.")
            self.dataset_test = dataset
        else:
            logging.error("Stage must be one of: fit, validate, test")
            return


    # Override
    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          collate_fn=self.dataset_train.collate_fn,
                          num_workers=self.num_data_workers,
                          shuffle=True, pin_memory=True)


    # Override
    def val_dataloader(self):
        return DataLoader(self.dataset_dev,
                          batch_size=self.batch_size,
                          collate_fn=self.dataset_dev.collate_fn,
                          num_workers=self.num_data_workers,
                          shuffle=False, pin_memory=True)


    # Override
    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                          batch_size=self.batch_size,
                          collate_fn=self.dataset_test.collate_fn,
                          num_workers=self.num_data_workers,
                          shuffle=False, pin_memory=True)


