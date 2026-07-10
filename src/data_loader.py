import torch
import glob
import os

class Dataload(torch.utils.data.Dataset):
    def __init__(self, processed_data_dir):
        # create search pattern with processed data directoy path
        search = os.path.join(processed_data_dir, '*.pt')

        # get all files in the processed data directory
        self.file_paths = glob.glob(search)

    def __len__(self):
        # return the length of list of files
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # get file indicated by the index
        file = self.file_paths[idx]

        # load the tensor from the provided file and return as (inputs, labels) pair
        data_tensor = torch.load(file, weights_only=False)
        return (data_tensor, data_tensor)
    
def get_dataloader(processed_data_dir, batch_size=16, shuffle=True):
    # create a dataset using Dataload with the files in the processed data directory
    dataset = Dataload(processed_data_dir)

    # return a DataLoader with the dataset and provided parameters
    return torch.utils.data.DataLoader (dataset, batch_size=batch_size, shuffle=shuffle) 

if __name__ == "__main__":
    processed_dir = "../data/processed"

    if os.path.exists(processed_dir):
        dataloader = get_dataloader(processed_dir)
        print (f"Batches in dataloaders: {len(dataloader)}")

        for inputs, targets in dataloader:
            print (f"Batch inputs shape: {inputs.shape}")
            break
    else:
        print ("No processed data directory found. Run data_processing.py")