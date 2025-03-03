import torch.utils.data
import pandas as pd


class UCFRep(torch.utils.data.Dataset):
    def __init__(self,
                 split="test",
                 data_dir = "data/UCFRep/UCF-101"):
        
        self.data_dir = data_dir
        self.split = split # set the split to load
        csv_path = f"datasets/ucf-rep/new_{self.split}.csv"
        self.df = pd.read_csv(csv_path)
        print(f"--- Loaded: {len(self.df)} videos for {self.split} --- " )


    def __getitem__(self, index):
        video_name = f"{self.data_dir}/{self.df.iloc[index]['name'][8:]}"
        count = f"{self.df.iloc[index]['counts']}"
      
        return video_name, float(count)    
    def __len__(self):
        return len(self.df)
    

    
        
    
