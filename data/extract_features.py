import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from scipy.integrate import simps
import mne
import entropy

def calculate_duration(t0, t1):
    return t1 - t0

def calculate_max_magnitude(data):
    magnitudes = np.abs(data)
    return np.max(magnitudes)

def calculate_mean_magnitude(data):
    magnitudes = np.abs(data)
    return np.mean(magnitudes)

def compute_skew(data):
    return stats.skew(data)

def compute_kurtosis(data):
    return stats.kurtosis(data)

def computer_emg_bandpowers(data, fs):
    freq_bands = [(10, 33), (33, 56), (56, 79), (79, 102), (102, 125)]
    psds, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=fs, fmin=0, fmax=int(fs/2), n_jobs=1, adaptive=True)
    psds = 10*np.log10(psds)
    band_pows = []
    freq_res = freqs[1] - freqs[0]
    for lo, hi in freq_bands:
        idx_band = np.logical_and(freqs >= lo, freqs <= hi)
        bp = simps(psds[idx_band], dx=freq_res)
        band_pows.append(bp)
    return band_pows

def computer_eeg_bandpowers(data, fs):
    freq_bands = [(1, 4), (4, 7), (7, 12), (12, 30)]    
    psds, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=fs, fmin=0, fmax=int(fs/2), n_jobs=1, adaptive=True)
    psds = 10*np.log10(psds)
    band_pows = []
    freq_res = freqs[1] - freqs[0]
    for lo, hi in freq_bands:
        idx_band = np.logical_and(freqs >= lo, freqs <= hi)
        bp = simps(psds[idx_band], dx=freq_res)
        band_pows.append(bp)
    return band_pows

def computer_eog_bandpowers(data, fs):
    freq_bands = [(0.5, 4), (4, 10)]    
    psds, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=fs, fmin=0, fmax=int(fs/2), n_jobs=1, adaptive=True)
    psds = 10*np.log10(psds)
    band_pows = []
    freq_res = freqs[1] - freqs[0]
    for lo, hi in freq_bands:
        idx_band = np.logical_and(freqs >= lo, freqs <= hi)
        bp = simps(psds[idx_band], dx=freq_res)
        band_pows.append(bp)
    return band_pows

def compute_top5_freqs_emg(data, fs, fmax=125):
    freq_bands = [(i, i+1) for i in range(10, fmax)]    
    psds, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=fs, fmin=0, fmax=fmax, n_jobs=1, adaptive=True)
    psds = 10*np.log10(psds)
    band_pows = []
    freq_res = freqs[1] - freqs[0]
    for lo, hi in freq_bands:
        idx_band = np.logical_and(np.round(freqs) >= lo, np.round(freqs) <= hi)
        bp = simps(psds[idx_band], dx=freq_res)
        band_pows.append((lo, hi, bp))
    sorted_freqs = sorted(band_pows, key = lambda x: x[2], reverse=True)
    top5_freqs = [tup[0] for tup in sorted_freqs[:5]]
    f1, f2, f3, f4, f5 = top5_freqs
    return f1, f2, f3, f4, f5

def compute_top5_freqs_eeg(data, fs, fmax=30):
    freq_bands = [(i, i+1) for i in range(1, fmax)]    
    psds, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=fs, fmin=0, fmax=fmax, n_jobs=1, adaptive=True)
    psds = 10*np.log10(psds)
    band_pows = []
    freq_res = freqs[1] - freqs[0]
    for lo, hi in freq_bands:
        idx_band = np.logical_and(np.round(freqs) >= lo, np.round(freqs) <= hi)
        bp = simps(psds[idx_band], dx=freq_res)
        band_pows.append((lo, hi, bp))
    sorted_freqs = sorted(band_pows, key = lambda x: x[2], reverse=True)
    top5_freqs = [tup[0] for tup in sorted_freqs[:5]]
    f1, f2, f3, f4, f5 = top5_freqs
    return f1, f2, f3, f4, f5

def compute_top5_freqs_eog(data, fs, fmax=10):
    freq_bands = [(i, i+1) for i in range(0, 10)]    
    psds, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=fs, fmin=0, fmax=fmax, n_jobs=1, adaptive=True)
    psds = 10*np.log10(psds)
    band_pows = []
    freq_res = freqs[1] - freqs[0]
    for lo, hi in freq_bands:
        idx_band = np.logical_and(np.round(freqs) >= lo, np.round(freqs) <= hi)
        bp = simps(psds[idx_band], dx=freq_res)
        band_pows.append((lo, hi, bp))
    sorted_freqs = sorted(band_pows, key = lambda x: x[2], reverse=True)
    top5_freqs = [tup[0] for tup in sorted_freqs[:5]]
    f1, f2, f3, f4, f5 = top5_freqs
    return f1, f2, f3, f4, f5

def compute_high_freq_contractions(data, fs, freq):
    f0 = freq
    Q = f0/5

    # Establish peak filter
    b, a = signal.iirpeak(f0, Q, fs)
    y = signal.filtfilt(b, a, data)
    
    # get frequency responses
    freq, h = signal.freqz(b, a, fs=fs)
    p = signal.find_peaks(y)[0]
    return len(p)

def compute_amplitude_percentiles(data, percentiles=[25, 50, 75, 90]):
    percentile_values = np.percentile(np.abs(data), percentiles)
    return percentile_values

def compute_zero_crossing_rate(data, fs):
    duration = len(data)/fs
    return len(np.nonzero(np.diff(data > 0))[0])/duration

def compute_root_mean_square(data):
    return np.sqrt(np.mean(data**2))

def clip_data(data, clipping_amplitude, clipped_value):
    data = np.array(data)
    data[np.where(np.abs(data) > clipping_amplitude)] = clipped_value
    return data

def calculate_emg_statistics(channel1, channel2, event_time_stamps, onset_times, fs):
    X_emg = []
    
    # Clip excessive amplitudes that are most likely from signal noise
    channel1 = clip_data(channel1, 1000, 0)
    channel2 = clip_data(channel2, 1000, 0)
    
    with mne.utils.use_log_level('error'):
        for i in range(len(event_time_stamps)):
            t0, t1 = event_time_stamps[i]
            segment_ch1, segment_ch2 = channel1[int(fs*t0):int(fs*t1)], channel2[int(fs*t0):int(fs*t1)]

            duration = calculate_duration(t0, t1)

            max_mag_channel1 = calculate_max_magnitude(segment_ch1)
            max_mag_channel2 = calculate_max_magnitude(segment_ch2)

            mean_mag_channel1 = calculate_mean_magnitude(segment_ch1)
            mean_mag_channel2 = calculate_mean_magnitude(segment_ch2)

            pf1_1, pf1_2, pf1_3, pf1_4, pf1_5 = compute_top5_freqs_emg(segment_ch1, fs, fmax=int(fs/2))
            pf2_1, pf2_2, pf2_3, pf2_4, pf2_5 = compute_top5_freqs_emg(segment_ch2, fs, fmax=int(fs/2))

            onset = onset_times[i]

            peak_freq_contractions_c1 = compute_high_freq_contractions(segment_ch1, fs, pf1_1)
            peak_freq_contractions_c2 = compute_high_freq_contractions(segment_ch2, fs, pf2_1)

            channel1_skew = compute_skew(segment_ch1)
            channel2_skew = compute_skew(segment_ch2)

            channel1_kurtosis = compute_kurtosis(segment_ch1)
            channel2_kurtosis = compute_kurtosis(segment_ch2)

            band1_c1, band2_c1, band3_c1, band4_c1, band5_c1 = computer_emg_bandpowers(segment_ch1, fs)
            band1_c2, band2_c2, band3_c2, band4_c2, band5_c2 = computer_emg_bandpowers(segment_ch2, fs)
            
            var_c1 = np.var(segment_ch1)
            var_c2 = np.var(segment_ch2)
            std_dev_1 = np.std(segment_ch1)
            std_dev_2 = np.std(segment_ch2)
            abs_std_dev_1 = np.std(np.abs(segment_ch1))
            abs_std_dev_2 = np.std(np.abs(segment_ch2))
            rms_1 = compute_root_mean_square(segment_ch1)
            rms_2 = compute_root_mean_square(segment_ch2)
            
            p25_1, p50_1, p75_1, p90_1 = compute_amplitude_percentiles(segment_ch1, percentiles=[25, 50, 75, 90])
            p25_2, p50_2, p75_2, p90_2 = compute_amplitude_percentiles(segment_ch2, percentiles=[25, 50, 75, 90])
            
            zcr_1 = compute_zero_crossing_rate(segment_ch1, fs)
            zcr_2 = compute_zero_crossing_rate(segment_ch2, fs)
            
            dfa_hurst_param_1 = entropy.detrended_fluctuation(segment_ch1)
            dfa_hurst_param_2 = entropy.detrended_fluctuation(segment_ch2)
            
            petrosian_fd_1 = entropy.petrosian_fd(segment_ch1)
            petrosian_fd_2 = entropy.petrosian_fd(segment_ch2)
            
            approx_entropy_1 = entropy.app_entropy(segment_ch1)
            approx_entropy_2 = entropy.app_entropy(segment_ch2)
            
            spectral_entropy_1 = entropy.spectral_entropy(segment_ch1, sf=fs, normalize=True)
            spectral_entropy_1 = spectral_entropy_1 if not(np.isnan(spectral_entropy_1)) else 0
            spectral_entropy_2 = entropy.spectral_entropy(segment_ch2, sf=fs, normalize=True)
            spectral_entropy_2 = spectral_entropy_2 if not(np.isnan(spectral_entropy_2)) else 0       

            xi = [duration, onset, mean_mag_channel1, mean_mag_channel2, max_mag_channel1, max_mag_channel2, \
                  pf1_1, pf1_2, pf1_3, pf1_4, pf1_5, \
                  pf2_1, pf2_2, pf2_3, pf2_4, pf2_5, \
                  peak_freq_contractions_c1, peak_freq_contractions_c2, \
                  band1_c1, band2_c1, band3_c1, band4_c1, band5_c1, \
                  band1_c2, band2_c2, band3_c2, band4_c2, band5_c2, \
                  channel1_skew, channel2_skew, channel1_kurtosis, channel2_kurtosis, \
                  var_c1, var_c2, \
                  p25_1, p50_1, p75_1, p90_1, p25_2, p50_2, p75_2, p90_2, \
                  zcr_1, zcr_2, \
                  std_dev_1, std_dev_2, abs_std_dev_1, abs_std_dev_2, rms_1, rms_2, \
                  dfa_hurst_param_1, dfa_hurst_param_2, petrosian_fd_1, petrosian_fd_2, \
                  approx_entropy_1, approx_entropy_2, spectral_entropy_1, spectral_entropy_2]

            X_emg.append(xi)
            
    feature_key = ['Event Duration (s)', 'Perceptible Onset Time (s)', 'Mean Absolute Amplitude - Channel 1 (uV)', 'Mean Absolute Amplitude - Channel 2(uV)', \
                   'Max Absolute Amplitude - Channel 1 (uV)', 'Max Absolute Amplitude - Channel 2 (uV)', \
                   'Peak Frequency 1 - Channel 1 (Hz)', 'Peak Frequency 2 - Channel 1 (Hz)', 'Peak Frequency 3 - Channel 1 (Hz)', \
                   'Peak Frequency 4 - Channel 1 (Hz)', 'Peak Frequency 5 - Channel 1 (Hz)', \
                   'Peak Frequency 1 - Channel 2 (Hz)', 'Peak Frequency 2 - Channel 2 (Hz)', 'Peak Frequency 3 - Channel 2 (Hz)', \
                   'Peak Frequency 4 - Channel 2 (Hz)', 'Peak Frequency 5 - Channel 2 (Hz)', \
                   'Peak Frequency Contractions - Channel 1', 'Peak Frequency Contractions - Channel 2', \
                   '10-33Hz Bandpower - Channel 1 (dB)', '33-56Hz Bandpower - Channel 1 (dB)', '56-79Hz Bandpower - Channel 1 (dB)', \
                   '79-102Hz Bandpower - Channel 1 (dB)', '102-125Hz Bandpower - Channel 1 (dB)', \
                   '10-33Hz Bandpower - Channel 2 (dB)', '33-56Hz Bandpower - Channel 2 (dB)', '56-79Hz Bandpower - Channel 2 (dB)', \
                   '79-102Hz Bandpower - Channel 2 (dB)', '102-125Hz Bandpower - Channel 2 (dB)', \
                   'Amplitude Skew - Channel 1', 'Amplitude Skew - Channel 2', \
                   'Amplitude Kurtosis - Channel 1', 'Amplitude Kurtosis - Channel 2', \
                   'Variance - Channel 1', 'Variance - Channel 2', '25th Percentile Absolute Amplitude - Channel 1', '50th Percentile Absolute Amplitude - Channel 1', \
                   '75th Percentile Absolute Amplitude - Channel 1', '90th Percentile Absolute Amplitude - Channel 1', '25th Percentile Absolute Amplitude - Channel 2', \
                   '50th Percentile Absolute Amplitude - Channel 2', '75th Percentile Absolute Amplitude - Channel 2', '90th Percentile Absolute Amplitude - Channel 2', \
                   'Zero Crossing Rate - Channel 1', 'Zero Crossing Rate - Channel 2', 'Standard Deviation - Channel 1', 'Standard Deviation - Channel 2', \
                   'Abolute Amplitude Standard Deviation - Channel 1', 'Abolute Amplitude Standard Deviation - Channel 2', 'Root Mean Square - Channel 1', \
                   'Root Mean Square - Channel 2', 'Detrended Fluctuation Hurst Parameter - Channel 1', 'Detrended Fluctuation Hurst Parameter - Channel 2', \
                   'Petrosian Fractal Dimension - Channel 1', 'Petrosian Fractal Dimension - Channel 2', 'Approximate Entropy - Channel 1', 'Approximate Entropy - Channel 2', \
                   'Spectral Entropy - Channel 1', 'Spectral Entropy - Channel 2']
    
    return X_emg, feature_key


def calculate_eeg_statistics(channel1, channel2, event_time_stamps, fs):
    X_eeg = []
    
    # Clip excessive amplitudes that are most likely from signal noise
    channel1 = clip_data(channel1, 1000, 0)
    channel2 = clip_data(channel2, 1000, 0)
    
    with mne.utils.use_log_level('error'):
        for i in range(len(event_time_stamps)):
            t0, t1 = event_time_stamps[i]
            segment_ch1 = channel1[int(fs*t0):int(fs*t1)]
            segment_ch2 = channel2[int(fs*t0):int(fs*t1)]

            max_mag_channel1 = calculate_max_magnitude(segment_ch1)
            max_mag_channel2 = calculate_max_magnitude(segment_ch2)

            mean_mag_channel1 = calculate_mean_magnitude(segment_ch1)
            mean_mag_channel2 = calculate_mean_magnitude(segment_ch2)

            pf1_1, pf1_2, pf1_3, pf1_4, pf1_5 = compute_top5_freqs_eeg(segment_ch1, fs, fmax=30)
            pf2_1, pf2_2, pf2_3, pf2_4, pf2_5 = compute_top5_freqs_eeg(segment_ch2, fs, fmax=30)

            channel1_skew = compute_skew(segment_ch1)
            channel2_skew = compute_skew(segment_ch2)

            channel1_kurtosis = compute_kurtosis(segment_ch1)
            channel2_kurtosis = compute_kurtosis(segment_ch2)

            band1_c1, band2_c1, band3_c1, band4_c1 = computer_eeg_bandpowers(segment_ch1, fs)
            band1_c2, band2_c2, band3_c2, band4_c2 = computer_eeg_bandpowers(segment_ch2, fs)
            
            var_c1 = np.var(segment_ch1)
            var_c2 = np.var(segment_ch2)
            std_dev_1 = np.std(segment_ch1)
            std_dev_2 = np.std(segment_ch2)
            abs_std_dev_1 = np.std(np.abs(segment_ch1))
            abs_std_dev_2 = np.std(np.abs(segment_ch2))
            rms_1 = compute_root_mean_square(segment_ch1)
            rms_2 = compute_root_mean_square(segment_ch2)
            
            p25_1, p50_1, p75_1, p90_1 = compute_amplitude_percentiles(segment_ch1, percentiles=[25, 50, 75, 90])
            p25_2, p50_2, p75_2, p90_2 = compute_amplitude_percentiles(segment_ch2, percentiles=[25, 50, 75, 90])
            
            zcr_1 = compute_zero_crossing_rate(segment_ch1, fs)
            zcr_2 = compute_zero_crossing_rate(segment_ch2, fs)
            
            dfa_hurst_param_1 = entropy.detrended_fluctuation(segment_ch1)
            dfa_hurst_param_2 = entropy.detrended_fluctuation(segment_ch2)
            
            petrosian_fd_1 = entropy.petrosian_fd(segment_ch1)
            petrosian_fd_2 = entropy.petrosian_fd(segment_ch2)
            
            approx_entropy_1 = entropy.app_entropy(segment_ch1)
            approx_entropy_2 = entropy.app_entropy(segment_ch2)
            
            spectral_entropy_1 = entropy.spectral_entropy(segment_ch1, sf=fs, normalize=True)
            spectral_entropy_1 = spectral_entropy_1 if not(np.isnan(spectral_entropy_1)) else 0
            spectral_entropy_2 = entropy.spectral_entropy(segment_ch2, sf=fs, normalize=True)
            spectral_entropy_2 = spectral_entropy_2 if not(np.isnan(spectral_entropy_2)) else 0        
            
            xi = [mean_mag_channel1, mean_mag_channel2, max_mag_channel1, max_mag_channel2, \
                  pf1_1, pf1_2, pf1_3, pf1_4, pf1_5, \
                  pf2_1, pf2_2, pf2_3, pf2_4, pf2_5, \
                  band1_c1, band2_c1, band3_c1, band4_c1, \
                  band1_c2, band2_c2, band3_c2, band4_c2, \
                  channel1_skew, channel2_skew, channel1_kurtosis, channel2_kurtosis, \
                  var_c1, var_c2, \
                  p25_1, p50_1, p75_1, p90_1, \
                  p25_2, p50_2, p75_2, p90_2, \
                  zcr_1, zcr_2, std_dev_1, std_dev_2, abs_std_dev_1, abs_std_dev_2, rms_1, rms_2, \
                  dfa_hurst_param_1, dfa_hurst_param_2, petrosian_fd_1, petrosian_fd_2, \
                  approx_entropy_1, approx_entropy_2, spectral_entropy_1, spectral_entropy_2]

            X_eeg.append(xi)
            
    feature_key = ['Mean Absolute Amplitude - Channel 1 (uV)', 'Mean Absolute Amplitude - Channel 2(uV)', \
                   'Max Absolute Amplitude - Channel 1 (uV)', 'Max Absolute Amplitude - Channel 2 (uV)', \
                   'Peak Frequency 1 - Channel 1 (Hz)', 'Peak Frequency 2 - Channel 1 (Hz)', 'Peak Frequency 3 - Channel 1 (Hz)', \
                   'Peak Frequency 4 - Channel 1 (Hz)', 'Peak Frequency 5 - Channel 1 (Hz)', \
                   'Peak Frequency 1 - Channel 2 (Hz)', 'Peak Frequency 2 - Channel 2 (Hz)', 'Peak Frequency 3 - Channel 2 (Hz)', \
                   'Peak Frequency 4 - Channel 2 (Hz)', 'Peak Frequency 5 - Channel 2 (Hz)', \
                   '1-4Hz Bandpower - Channel 1 (dB)', '4-7Hz Bandpower - Channel 1 (dB)', '7-12Hz Bandpower - Channel 1 (dB)', \
                   '12-302Hz Bandpower - Channel 1 (dB)', \
                   '1-4Hz Bandpower - Channel 2 (dB)', '4-7Hz Bandpower - Channel 2 (dB)', '7-12Hz Bandpower - Channel 2 (dB)', \
                   '12-302Hz Bandpower - Channel 2 (dB)', \
                   'Amplitude Skew - Channel 1', 'Amplitude Skew - Channel 2', \
                   'Amplitude Kurtosis - Channel 1', 'Amplitude Kurtosis - Channel 2', \
                   'Variance - Channel 1', 'Variance - Channel 2', '25th Percentile Absolute Amplitude - Channel 1', '50th Percentile Absolute Amplitude - Channel 1', \
                   '75th Percentile Absolute Amplitude - Channel 1', '90th Percentile Absolute Amplitude - Channel 1', '25th Percentile Absolute Amplitude - Channel 2', \
                   '50th Percentile Absolute Amplitude - Channel 2', '75th Percentile Absolute Amplitude - Channel 2', '90th Percentile Absolute Amplitude - Channel 2', \
                   'Zero Crossing Rate - Channel 1', 'Zero Crossing Rate - Channel 2', 'Standard Deviation - Channel 1', 'Standard Deviation - Channel 2', \
                   'Abolute Amplitude Standard Deviation - Channel 1', 'Abolute Amplitude Standard Deviation - Channel 2', 'Root Mean Square - Channel 1', \
                   'Root Mean Square - Channel 2', 'Detrended Fluctuation Hurst Parameter - Channel 1', 'Detrended Fluctuation Hurst Parameter - Channel 2', \
                   'Petrosian Fractal Dimension - Channel 1', 'Petrosian Fractal Dimension - Channel 2', 'Approximate Entropy - Channel 1', 'Approximate Entropy - Channel 2', \
                   'Spectral Entropy - Channel 1', 'Spectral Entropy - Channel 2']
    
    return X_eeg, feature_key


def calculate_eog_statistics(channel1, channel2, event_time_stamps, fs):
    X_eog = []
    
    # Clip excessive amplitudes that are most likely from signal noise
    channel1 = clip_data(channel1, 1000, 0)
    channel2 = clip_data(channel2, 1000, 0)
    
    with mne.utils.use_log_level('error'):
        for i in range(len(event_time_stamps)):
            t0, t1 = event_time_stamps[i]
            segment_ch1 = channel1[int(fs*t0):int(fs*t1)]
            segment_ch2 = channel2[int(fs*t0):int(fs*t1)]

            max_mag_channel1 = calculate_max_magnitude(segment_ch1)
            max_mag_channel2 = calculate_max_magnitude(segment_ch2)

            mean_mag_channel1 = calculate_mean_magnitude(segment_ch1)
            mean_mag_channel2 = calculate_mean_magnitude(segment_ch2)

            pf1_1, pf1_2, pf1_3, pf1_4, pf1_5 = compute_top5_freqs_eog(segment_ch1, fs, fmax=30)
            pf2_1, pf2_2, pf2_3, pf2_4, pf2_5 = compute_top5_freqs_eog(segment_ch2, fs, fmax=30)

            channel1_skew = compute_skew(segment_ch1)
            channel2_skew = compute_skew(segment_ch2)

            channel1_kurtosis = compute_kurtosis(segment_ch1)
            channel2_kurtosis = compute_kurtosis(segment_ch2)

            band1_c1, band2_c1 = computer_eog_bandpowers(segment_ch1, fs)
            band1_c2, band2_c2 = computer_eog_bandpowers(segment_ch2, fs)
            
            var_c1 = np.var(segment_ch1)
            var_c2 = np.var(segment_ch2)
            std_dev_1 = np.std(segment_ch1)
            std_dev_2 = np.std(segment_ch2)
            abs_std_dev_1 = np.std(np.abs(segment_ch1))
            abs_std_dev_2 = np.std(np.abs(segment_ch2))
            rms_1 = compute_root_mean_square(segment_ch1)
            rms_2 = compute_root_mean_square(segment_ch2)
            
            p25_1, p50_1, p75_1, p90_1 = compute_amplitude_percentiles(segment_ch1, percentiles=[25, 50, 75, 90])
            p25_2, p50_2, p75_2, p90_2 = compute_amplitude_percentiles(segment_ch2, percentiles=[25, 50, 75, 90])
            
            zcr_1 = compute_zero_crossing_rate(segment_ch1, fs)
            zcr_2 = compute_zero_crossing_rate(segment_ch2, fs)
            
            dfa_hurst_param_1 = entropy.detrended_fluctuation(segment_ch1)
            dfa_hurst_param_2 = entropy.detrended_fluctuation(segment_ch2)
            
            petrosian_fd_1 = entropy.petrosian_fd(segment_ch1)
            petrosian_fd_2 = entropy.petrosian_fd(segment_ch2)
            
            approx_entropy_1 = entropy.app_entropy(segment_ch1)
            approx_entropy_2 = entropy.app_entropy(segment_ch2)
            
            spectral_entropy_1 = entropy.spectral_entropy(segment_ch1, sf=fs, normalize=True)
            spectral_entropy_1 = spectral_entropy_1 if not(np.isnan(spectral_entropy_1)) else 0
            spectral_entropy_2 = entropy.spectral_entropy(segment_ch2, sf=fs, normalize=True)
            spectral_entropy_2 = spectral_entropy_2 if not(np.isnan(spectral_entropy_2)) else 0
            
            eog1_initial_seg = channel1[int(fs*t0):int(fs*(t0+0.3))]
            eog2_initial_seg = channel2[int(fs*t0):int(fs*(t0+0.3))]
            eog1_initial_deflection = np.clip(eog1_initial_seg[-1]-eog1_initial_seg[0], -100, 100)
            eog2_initial_deflection = np.clip(eog2_initial_seg[-1]-eog2_initial_seg[0], -100, 100)
            eog1_initial_deflection_sign = 1 if eog1_initial_deflection > 0 else -1
            eog2_initial_deflection_sign = 1 if eog2_initial_deflection > 0 else -1
            
            xi = [mean_mag_channel1, mean_mag_channel2, max_mag_channel1, max_mag_channel2, \
                  pf1_1, pf1_2, pf1_3, pf1_4, pf1_5, \
                  pf2_1, pf2_2, pf2_3, pf2_4, pf2_5, \
                  band1_c1, band2_c1, \
                  band1_c2, band2_c2, \
                  channel1_skew, channel2_skew, channel1_kurtosis, channel2_kurtosis, \
                  var_c1, var_c2, \
                  p25_1, p50_1, p75_1, p90_1, \
                  p25_2, p50_2, p75_2, p90_2, \
                  zcr_1, zcr_2, std_dev_1, std_dev_2, abs_std_dev_1, abs_std_dev_2, rms_1, rms_2, \
                  dfa_hurst_param_1, dfa_hurst_param_2, petrosian_fd_1, petrosian_fd_2, \
                  approx_entropy_1, approx_entropy_2, spectral_entropy_1, spectral_entropy_2, \
                  eog1_initial_deflection, eog2_initial_deflection, eog1_initial_deflection_sign, eog2_initial_deflection_sign]

            X_eog.append(xi)
            
    feature_key = ['Mean Absolute Amplitude - Channel 1 (uV)', 'Mean Absolute Amplitude - Channel 2(uV)', \
                   'Max Absolute Amplitude - Channel 1 (uV)', 'Max Absolute Amplitude - Channel 2 (uV)', \
                   'Peak Frequency 1 - Channel 1 (Hz)', 'Peak Frequency 2 - Channel 1 (Hz)', 'Peak Frequency 3 - Channel 1 (Hz)', \
                   'Peak Frequency 4 - Channel 1 (Hz)', 'Peak Frequency 5 - Channel 1 (Hz)', \
                   'Peak Frequency 1 - Channel 2 (Hz)', 'Peak Frequency 2 - Channel 2 (Hz)', 'Peak Frequency 3 - Channel 2 (Hz)', \
                   'Peak Frequency 4 - Channel 2 (Hz)', 'Peak Frequency 5 - Channel 2 (Hz)', \
                   '0.5-4Hz Bandpower - Channel 1 (dB)', '4-10Hz Bandpower - Channel 1 (dB)', \
                   '0.5-4Hz Bandpower - Channel 2 (dB)', '4-10Hz Bandpower - Channel 2 (dB)', \
                   'Amplitude Skew - Channel 1', 'Amplitude Skew - Channel 2', \
                   'Amplitude Kurtosis - Channel 1', 'Amplitude Kurtosis - Channel 2', \
                   'Variance - Channel 1', 'Variance - Channel 2', '25th Percentile Absolute Amplitude - Channel 1', '50th Percentile Absolute Amplitude - Channel 1', \
                   '75th Percentile Absolute Amplitude - Channel 1', '90th Percentile Absolute Amplitude - Channel 1', '25th Percentile Absolute Amplitude - Channel 2', \
                   '50th Percentile Absolute Amplitude - Channel 2', '75th Percentile Absolute Amplitude - Channel 2', '90th Percentile Absolute Amplitude - Channel 2', \
                   'Zero Crossing Rate - Channel 1', 'Zero Crossing Rate - Channel 2', 'Standard Deviation - Channel 1', 'Standard Deviation - Channel 2', \
                   'Abolute Amplitude Standard Deviation - Channel 1', 'Abolute Amplitude Standard Deviation - Channel 2', 'Root Mean Square - Channel 1', \
                   'Root Mean Square - Channel 2', 'Detrended Fluctuation Hurst Parameter - Channel 1', 'Detrended Fluctuation Hurst Parameter - Channel 2', \
                   'Petrosian Fractal Dimension - Channel 1', 'Petrosian Fractal Dimension - Channel 2', 'Approximate Entropy - Channel 1', 'Approximate Entropy - Channel 2', \
                   'Spectral Entropy - Channel 1', 'Spectral Entropy - Channel 2', 'Initial Deflection (Clipped) - Channel 1', 'Initial Deflection (Clipped) - Channel 2', \
                   'Deflection Sign - Channel 1', 'Deflection Sign - Channel 2']
    
    return X_eog, feature_key


fs = 250 # Sampling frequency

if __name__ == '__main__':
    signal_data_dir = './signal_data'
    timestamps_dir = './timestamps'
    feature_data_dir = './feature_data'
    subject_ids = ['subject{}'.format(i) for i in range(10)]
    session_ids = ['Morning', 'Evening']
    for subject_id in subject_ids:
        for session_id in session_ids:
            data_fname = '{}_{}_separated_signals.csv'.format(subject_id, session_id)
            signal_data_fpath = os.path.join(signal_data_dir, subject_id, data_fname)
            
            timestamps_fname = '{}_{}_timestamps.csv'.format(subject_id, session_id)
            timestamps_fpath = os.path.join(timestamps_dir, subject_id, timestamps_fname)
            
            data = pd.read_csv(signal_data_fpath)
            EEG = np.array([data['EEG Channel 1'], data['EEG Channel 2']])
            EOG = np.array([data['EOG Channel 1'], data['EOG Cannel 2']])
            EMG = np.array([data['EMG Channel 1'], data['EMG Channel 2']])
            event_timestamps_df = pd.read_csv(timestamps_fpath)
            event_labels = np.array(event_timestamps_df['Task Label'])
            event_timestamps = event_timestamps_df[['Event Start (s)', 'Event Stop (s)']].to_numpy()
            perceptible_onset_times = event_timestamps_df['Perceptible Onset Time'].to_numpy()
            
            X_emg, emg_data_ids = calculate_emg_statistics(EMG[0], EMG[1], event_timestamps, perceptible_onset_times, fs)
            emg_data_ids = emg_data_ids[:2] + ['EMG {}'.format(feat_id) for feat_id in emg_data_ids[2:]]

            X_eeg, eeg_data_ids = calculate_eeg_statistics(EEG[0], EEG[1], event_timestamps, fs)
            eeg_data_ids = ['EEG {}'.format(feat_id) for feat_id in eeg_data_ids]

            X_eog, eog_data_ids = calculate_eog_statistics(EOG[0], EOG[1], event_timestamps, fs)
            eog_data_ids = ['EOG {}'.format(feat_id) for feat_id in eog_data_ids]

            data_ids = np.hstack([['Task Label'], emg_data_ids, eeg_data_ids, eog_data_ids])

            all_activities = np.hstack([event_labels.reshape(-1,1), X_emg, X_eeg, X_eog])

            df = pd.DataFrame(all_activities, columns=data_ids)
            if not os.path.isdir(os.path.join(feature_data_dir, subject_id)):
                os.mkdir(os.path.join(feature_data_dir, subject_id))
            feature_fname = '{}_{}_variables.csv'.format(subject_id, session_id)
            feature_fpath = os.path.join(feature_data_dir, subject_id, feature_fname)
            df.to_csv(feature_fpath, index=False)