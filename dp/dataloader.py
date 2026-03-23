from torch.utils.data import Dataset, DataLoader


class PromptDataset(Dataset):
    """
    PromptDataset class for loading the prompt from a text file.
    One line of the text file should be a single prompt.
    """

    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def get_dataloader(dataset_path, batch_size, shuffle, num_workers):
    """
    Creates a DataLoader for the PromptDataset class.

    Args:
        dataset_path (str): path to the text file containing the prompts.
        batch_size (int):  dimension of the batch.
        shuffle (bool): if True, the data will be shuffled.
        num_workers (int): umber of subprocess to use to load the data.

    Returns:
        DataLoader: a DataLoader for the prompts.
    """
    dataset = PromptDataset(dataset_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
