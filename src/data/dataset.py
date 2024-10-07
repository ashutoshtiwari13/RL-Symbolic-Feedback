from datasets import load_dataset
from transformers import AutoTokenizer
from trl.core import LengthSampler

def build_dataset(config, dataset_name="stanfordnlp/imdb", input_min_text_length=2, input_max_text_length=8, max_samples=10000):
    """
    Build and preprocess the dataset for training.
    
    Args:
        config (PPOConfig): Configuration object.
        dataset_name (str): Name of the dataset to load.
        input_min_text_length (int): Minimum input text length.
        input_max_text_length (int): Maximum input text length.
        max_samples (int): Maximum number of samples to use.
    
    Returns:
        Dataset: Preprocessed dataset.
        AutoTokenizer: Tokenizer used for preprocessing.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(dataset_name, split="train")
    ds = ds.select(range(min(len(ds), max_samples)))
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        inputs = tokenizer(
            sample["review"],
            padding="max_length",
            truncation=True,
            max_length=input_size(),
            return_tensors="pt"
        )
        sample["input_ids"] = inputs["input_ids"].squeeze()
        sample["attention_mask"] = inputs["attention_mask"].squeeze()
        sample["query"] = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "query"])
    return ds, tokenizer

def build_dataset_standard(config, dataset_name="stanfordnlp/imdb", input_min_text_length=2, input_max_text_length=8, max_samples=10000):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(dataset_name, split="train")
    ds = ds.select(range(min(len(ds), max_samples)))
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        inputs = tokenizer(
            sample["review"],
            padding="max_length",
            truncation=True,
            max_length=input_size(),
            return_tensors="pt"
        )
        sample["input_ids"] = inputs["input_ids"].squeeze()
        sample["query"] = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch", columns=["input_ids", "query"])
    return ds, tokenizer


def collator(data):
    """
    Collate function for batching data.
    
    Args:
        data (List[Dict]): List of samples to collate.
    
    Returns:
        Dict: Collated batch of samples.
    """
    return dict((key, [d[key] for d in data]) for key in data[0])