import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
# import seaborn as sns
import speechpy
#import librosa
from statsmodels.tsa.ar_model import AR
from scipy.fftpack import fft, ifft
from pywt import wavedec, waverec


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
   
    def generate_features(self, psec, rawwave, low_freq=0, high_freq=30, mode=1, filtered=False, wavt=False, AR_order=10):
        #first column is time column and should not be selected --> the column index starts from 1
        psec = psec.iloc[:, int(low_freq * 4 + 1) : int(high_freq * 4 + 1 + 1)] #four columns per hertz
        vector = self.compress(psec)
        
        if mode == 1:  #mfcc + power spectrum
            mfcc = speechpy.feature.mfcc(rawwave, sampling_frequency=512, fft_length=256, frame_length=0.3, frame_stride=0.05, low_frequency=0.1)
            vector = np.concatenate((vector, self.compress(mfcc)), axis=1)
        if mode == 2: #AR + power spectrum
            AR_params = self.generate_AR_para(rawwave, filtered=filtered, wavt=wavt, AR_order=AR_order)
            AR_params = np.array(AR_params).reshape(1, len(AR_params))
            vector = np.concatenate((vector, AR_params), axis=1)
        if mode == 3: #AR
            AR_params = self.generate_AR_para(rawwave, filtered=filtered, wavt=wavt, AR_order=AR_order)
            AR_params = np.array(AR_params).reshape(1, len(AR_params))
            vector = AR_params
        if mode == 4: #mfcc + AR
            mfcc = speechpy.feature.mfcc(rawwave, sampling_frequency=512, fft_length=256, frame_length=0.3, frame_stride=0.05, low_frequency=1.5)
            AR_params = self.generate_AR_para(rawwave, filtered=filtered, wavt=wavt, AR_order=AR_order)
            AR_params = np.array(AR_params).reshape(1, len(AR_params))
            vector = np.concatenate((self.compress(mfcc), AR_params), axis=1)
        if mode ==5: #mfcc
            mfcc = speechpy.feature.mfcc(rawwave, sampling_frequency=512, fft_length=256, frame_length=0.3, frame_stride=0.05, low_frequency=1.5)
            vector = self.compress(mfcc)
            
        return vector
    
    def compress(self,arr):
        # Get rid of temporal dimension by taking a median of each column
        # (also reshape for sklearn's pairwise distance metrics)
        medians = np.median(arr, axis=0).reshape(1,-1)
        p_25 = np.percentile(arr, q=25, axis=0).reshape(1, -1)
        p_75 = np.percentile(arr, q=75, axis=0).reshape(1, -1)
        
        return np.concatenate((medians, p_25, p_75), axis=1)
    
    def selective_freq_range(self, signal, high_freq=None, low_freq=0):
        #filter the original signal with selective frequency range
        N = len(signal) #number of samples
        fs = 512 #sampling frequency
        psd = fft(signal) #power spectral density, derived from fast Fourier transform
        
        #frequency band
        W = []
        for i in range(N):
            #if N is odd
            if N % 2 != 0:
                flip_val = (N-1)/2
                if i < flip_val + 1:
                    W.append(i * fs / N)
                else:
                    W.append( (i - 2 * flip_val - 1) * fs / N  )
            else:
                #if N is even
                flip_val = N/2 - 1
                if i < flip_val + 1:
                    W.append(i * fs / N)
                else:
                    W.append( (i - 2 * flip_val - 1) * fs / N)
        #convert list into an array
        W = np.array(W)
        
        filtered_psd = psd.copy()
        if high_freq != None:
            filtered_psd[(W>high_freq)] = 0
        filtered_psd[(W<low_freq)] = 0
        filtered_signal = np.abs(ifft(filtered_psd)) #取模
        
        return filtered_signal, W, filtered_psd
    
    def plot_psd(self, psd, N, fs, name, high_freq):
        '''
        Plotting the power spectral density of EEG signal
        high_freq: the high frequency band edge
        N: sample number
        fs: sampling frequency
        name: file name to be saved
        '''
        title_names = ["beach", "finger", "rest"]
        for title_name in title_names:
            if title_name in name:
                title = title_name
                
        plt.figure()
        power = 2.0/N * np.abs(psd[0 : high_freq * N // fs])
        xf = np.linspace(0.0, high_freq, len(power))
        plt.plot(xf, power)
        plt.xlabel("frequency Hz")
        plt.ylabel("Power Spectral Density")
        plt.title(title)
        plt.grid()
        plt.savefig("../plots/frequency/" + name + ".png")
        #plt.show()
        plt.close()
    
    def wavelet_transform(self, rawwave, wavelet='db4', de_level=5, level_required=5): 
        rawwave,_,_ = self.selective_freq_range(rawwave, low_freq=2.0)
        #multi-level discrete wavelet transform
        coeffs = wavedec(rawwave, wavelet, level=de_level)
        #A5, D5, D4, D3, D2, D1 = coeffs
        
        if type(level_required) == int:
            levels = list(range(de_level))
            levels.remove(de_level+1 - level_required)
            levels.remove(0)
        else: #level_required is a list of required levels
            for level in level_required:
                levels = list(range(de_level))
                levels.remove(de_level+1 - level)
            levels.remove(0)
                
        
        for i in levels:
            coeffs[i] = np.zeros_like(coeffs[i])
        
        reconst_signal = waverec(coeffs, wavelet)
        
        return reconst_signal, rawwave
        
        
    def generate_AR_para(self, rawwave, filtered=False, wavt=False, AR_order=10):
        signal = rawwave
                    
        '''
        W = fftfreq(signal.size, d= 1 / 512)
        psd = rfft(signal) #discrete Fourier transform of a real sequence
        filtered_psd = psd.copy()
        filtered_psd[(W<30)] = 0
        filtered_signal = irfft(filtered_psd)
        '''
        if filtered == True:
            if wavt == False:
                filtered_signal, _, _ = self.selective_freq_range(signal, high_freq=30, low_freq=1.5)
                ARModel = AR(filtered_signal)
            else:
                filtered_signal, _ = self.wavelet_transform(signal)
                ARModel = AR(filtered_signal)
        else:
            ARModel = AR(signal)
        
        #ARModel_fit = ARModel.fit()
        ARModel_fit = ARModel.fit(maxlag=AR_order)
        
        return ARModel_fit.params

def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))


def generate_data(task='beach', mode=5):
    
    targetstr = '../Experimental/' + task
    str2bdeleted = "../Experimental/" + task + "\\testsubject"
    folders = listdir_nohidden(targetstr)
        
    labels = []
    data = []
    #AR_params = []
    PD = process_data()
    #reconstr_datas = []
    #original_datas = []
    for folder in folders:
        sub_folder = listdir_nohidden(folder)[0]
        print(sub_folder)
        CD = collect_data(sub_folder)
        psec_list = CD.readPowerSpec()
        rwave_list = CD.readFilteredWave()
        label = folder.replace(str2bdeleted, "")
        for i in range(len(psec_list)):
            feature_vec = PD.generate_features(psec_list[i], rwave_list[i].iloc[:,1].values, low_freq = 1.5, mode=mode, filtered=True, wavt=True)
            #AR_param = PD.generate_AR_para(rwave_list[i].iloc[:,1].values, filtered=False)
            labels.append(float(label))
            data.append(feature_vec[0])
            #red, ori = PD.wavelet_transform(rwave_list[i].iloc[:,1].values)
            #reconstr_datas.append(red)
            #original_datas.append(ori)
            
            #AR_params.append(AR_param)
    
    #convert list into pandas dataframe
    #data = pd.DataFrame({"data":data, "subject":labels, "task": task})
    #output as csv file
    #data.to_csv("Data" + task + ".csv", index=False)
    
    train_x = np.array(data)
    train_y = np.array(labels).reshape(len(labels),1)
    
    data = np.concatenate((train_x, train_y), axis=1)
    
    #AR_x = np.array(AR_params)
    #AR_data = np.concatenate((AR_x, train_y), axis=1)
    np.savetxt(task + "M.csv", data, delimiter=",")
    

if __name__=="__main__":
    
    #generate_data() 
    task = "rest"
    person = "1"
    targetstr = '../Experimental/' + task
    str2bdeleted = "../Experimental/" + task + "\\testsubject"
    folders = listdir_nohidden(targetstr)
    
    PD = process_data()
    rwave_dict = {}

    for folder in folders:
        sub_folder = listdir_nohidden(folder)[0]
        print(sub_folder)
        CD = collect_data(sub_folder)
        rwave_list = CD.readFilteredWave()
        subject = folder.replace(str2bdeleted, "")
        rwave_dict[subject] = rwave_list
    
    del rwave_list
    
    rwave_list = rwave_dict[person]
    count = 0
    fs = 512 #sampling frequency
    sec = 5
    freq_edge = 30
    for rwave in rwave_list:
        rwave = rwave[' Value']
        rwave = rwave.iloc[: sec * fs]
        _, _, psd = PD.selective_freq_range(rwave, high_freq=30)
        #len(psd): the number of samples
        PD.plot_psd(psd, len(psd), fs, person + " " + task + str(count), high_freq = freq_edge)
        count+=1
    
    
    '''
    ada = AdaBoostClassifier()
    ada.fit(train_x, train_y)
    y = ada.predict(train_x)
    '''
    #graph = confusion_matrix(train_y, y)

    '''
    raw_psec = psec_list[0]
    rwave_list = CD.readRawWave()
    rwave = rwave_list[0]
    
    PD = process_data()
    
    psec_vec = PD.generate_features(raw_psec, rwave.iloc[:,1].values)
    '''

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