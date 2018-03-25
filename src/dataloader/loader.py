import os
from src.dataloader.config import M, F

class DatasetLoader:
    def __init__(self):
        print("DatasetLoader initialized...")

    def get_data(self, DATA_DIR):
        subfolders = ['male', 'female']

        data = []
        target = []
        for subfolder in subfolders:

            files = os.listdir(os.path.join(DATA_DIR, subfolder))
            for file in files:

                # Handling for mac
                if file == '.DS_Store': continue

                with open(os.path.join(DATA_DIR, subfolder, file), encoding="latin-1") as f:
                    x = {}
                    for line in f:
                        line = line.rstrip()
                        tokens = line.split('\t')
                        x[tokens[0]] = float(tokens[1])

                    data.append(x)

                    target.append(M) if subfolder == 'male' else target.append(F)

        return data, target