from src.dataloader.config import PARTICIPANTS, M, F
import os

RAW_DATA_DIR = '../resources/word_frequency_by_conv/'
DATA_DIR = '../resources/dataset'

class PreProcessor(object):

    def __init__(self):
        print("Instantiating PreProcessor...")

    data = {M: {}, F: {}}

    def extractAndSeperateGenderWCList(self):
        subfolders = ['S%02d' % i for i in range(1, 24)]
        #subfolders.append('S0%d' % i for i in range(10, 24))

        for subfolder in subfolders:
            files = os.listdir(os.path.join(RAW_DATA_DIR, subfolder))
            for file in files:

                if 'all' in file or '.DS_Store' in file: continue

                filename = os.path.splitext(file)[0]
                self.__addFileToMemory(os.path.join(RAW_DATA_DIR, subfolder, file), filename);

        return self.data


    def __addFileToMemory(self, file, filename):

        participant = filename.split('-')[1]
        gender = PARTICIPANTS[filename.split('-')[1]]

        if participant not in self.data[gender]:
            self.data[gender][participant] = {}

        with open(file, encoding="latin-1") as f:

            for line in f:
                tokens = line.split('\t')
                self.data[gender][participant][tokens[0]] = self.data[gender][participant].get(tokens[0], 0.0) + float(tokens[1])


    #data - > data{ 0: {'P001' : {'I':9}}}

    def generateWordFreqByGender(self, data):
        # write data of male participants to files
        for participant, wordsFreq in data[M].items():
            with open(os.path.join(DATA_DIR, 'male', participant + '.txt'), 'w') as f:
                for word, freq in wordsFreq.items():
                    f.write(word + '\t' + str(freq) + '\n')

        # write data of male participants to files
        for participant, wordsFreq in data[F].items():
            with open(os.path.join(DATA_DIR, 'female', participant + '.txt'), 'w') as f:
                for word, freq in wordsFreq.items():
                    f.write(word + '\t' + str(freq) + '\n')