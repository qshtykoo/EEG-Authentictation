import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
# import seaborn as sns
import math
import speechpy
#import librosa



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

    def readFilteredWave(self):
        fwave_list = []
        csv_files = os.listdir(self.file_dir)
        targeted_str = "filtered"
        for csv_file in csv_files:
            if targeted_str in csv_file:
                csv_full_path = os.path.join(self.file_dir, csv_file)
                filtered_wave = pd.read_csv(csv_full_path)
                fwave_list.append(filtered_wave)

        return fwave_list


class process_data:
   
    def generate_features(self, psec, rawwave, low_freq=0, high_freq=30):
        #first column is time column and should not be selected --> the column index starts from 1
        psec = psec.iloc[:, low_freq * 4 + 1 : high_freq * 4 + 1 + 1] #four columns per hertz
        vector = self.compress(psec)
        
        mfcc = speechpy.feature.mfcc(rawwave, sampling_frequency=512, fft_length=256, frame_length=0.3, frame_stride=0.05, low_frequency=0.1)
        vector = np.concatenate((vector, self.compress(mfcc)), axis=1)
        
        return vector
    
    def compress(self,arr):
        # Get rid of temporal dimension by taking a median of each column
        # (also reshape for sklearn's pairwise distance metrics)
        medians = np.median(arr, axis=0).reshape(1,-1)
        p_25 = np.percentile(arr, q=25, axis=0).reshape(1, -1)
        p_75 = np.percentile(arr, q=75, axis=0).reshape(1, -1)
        
        return np.concatenate((medians, p_25, p_75), axis=1)

def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

if __name__=="__main__":

    targetstr = 'mlmps18_group01-master/Experimental/beach'
    folders = listdir_nohidden(targetstr)
    psec_l = []
    for folder in folders:
        print(folder)
        sub_folder = listdir_nohidden(folder)[0]
        print(sub_folder)
        CD = collect_data(sub_folder)
        psec_list = CD.readPowerSpec()
        psec_l.append(psec_list)


    raw_psec = psec_list[0]
    rwave_list = CD.readRawWave()
    rwave = rwave_list[0]
    
    PD = process_data()
    
    psec_vec = PD.generate_features(raw_psec, rwave.iloc[:,1].values)


    #header_list_filtered = PD.lowPass_header(30)
    
    #first_second = raw_psec.loc[0:7, :]
    #selected_time = np.sum(raw_psec.iloc[0:31, 1:122], axis =0)
    #plt.plot(header_list_filtered, selected_time)
    #plt.show()

    


    #rwave.plot()
    #fwave.plot()

    '''
    fig, axs = plt.subplots(2,1, constrained_layout=True)
    axs[0].plot(np.array(range(len(rwave))), rwave[" Value"])
    axs[0].set_title('Raw Wave Data')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Voltage')
    axs[0].set_ylim([-200, 200])
    fig.suptitle('Raw Wave Data in Time Domain')

    axs[1].plot(np.array(range(len(fwave))), fwave[" Value"])
    axs[1].set_title('Raw Filtered Wave Data')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Voltage')
    axs[1].set_ylim([-200, 200])

    plt.show()
    
    raw_psec = raw_psec.iloc[:, 1:256]
    ax = sns.heatmap(raw_psec)
    #xticks = np.arange(63.5+1)
    #ax.set_xticks(xticks*ax.get_xlim()[1]/(2*math.pi))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Power Spectrum Heatmap')
    plt.show()
    '''