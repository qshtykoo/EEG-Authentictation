import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os


class collect_data:

    def __init__(self, data_dir):
        self.file_dir = data_dir

    def readPowerSpec(self):
        psec_list = []
        csv_files = os.listdir(self.file_dir)
        targeted_str = "powerspec"
        for csv_file in csv_files:
            if targeted_str in csv_file:
                csv_full_path = os.path.join(self.file_dir, csv_file)
                raw_psec = pd.read_csv(csv_full_path)
                psec_list.append(raw_psec)

        return psec_list

    def readRawWave(self):
        rwave_list = []
        csv_files = os.listdir(self.file_dir)
        targeted_str = "rawwave"
        for csv_file in csv_files:
            if targeted_str in csv_file:
                csv_full_path = os.path.join(self.file_dir, csv_file)
                raw_wave = pd.read_csv(csv_full_path)
                rwave_list.append(raw_wave)

        return rwave_list

class process_data:

    def __init__(self, data_list):
        self.data_list = data_list

    def lowPass_header(self, freq):
        header_list = list(self.data_list)
        header_list = header_list[1:] #get rid of "time" column
        #convert string list to float list
        header_list = list(map(float, header_list))

        targeted_index = header_list.index(freq)
        index_list = np.array(range(targeted_index+1)) #filtered out frequencies above 30Hz
        header_list_filtered = [ header_list[index] for index in index_list ] #121 frequency bins
    
        return header_list_filtered
                

if __name__=="__main__":
    
    dir_path = 'testsubject1/2018-03-17/'
    CD = collect_data(dir_path)
    psec_list = CD.readPowerSpec()

    #pick a sample to visualize
    raw_psec = psec_list[0]
    PD = process_data(raw_psec)

    header_list_filtered = PD.lowPass_header(30)
    
    #first_second = raw_psec.loc[0:7, :]
    selected_time = np.sum(raw_psec.iloc[0:31, 1:122], axis =0)
    plt.plot(header_list_filtered, selected_time)
    plt.show()
