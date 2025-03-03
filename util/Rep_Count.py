import torch.utils.data
import pandas as pd

class Rep_Count(torch.utils.data.Dataset):
    def __init__(self,
                 split="test",
                 data_dir = "data/RepCount/LLSP/video"):
        
        self.data_dir = data_dir
        self.split = split 
        csv_path = f"datasets/repcount/{self.split}_with_fps.csv"
        self.df = pd.read_csv(csv_path)
        print(f"--- Loaded: {len(self.df)} videos for {self.split} --- " )

    def __getitem__(self, index):
         
        video_name = f"{self.data_dir}/{self.split}/{self.df.iloc[index]['name']}"
        count = f"{self.df.iloc[index]['count']}"
      
        return video_name, count
            

    def __len__(self):
        return len(self.df)
    
    
        
    
