import numpy as np
import matplotlib.pyplot as plt


class AudioAugmenter:
    def __init__(self, noise_level=0.01, num_masks=2, mask_prob=0.5, mask_range=(0.1, 0.3)):
        self.noise_level = noise_level
        self.num_masks = num_masks
        self.mask_prob = mask_prob
        self.mask_range = mask_range

    def time_stretch(self, time, amp, rate):
        new_length = int(len(time) * rate)
        new_time = np.linspace(time[0], time[-1], new_length)
        new_amp = np.interp(new_time, time, amp)
        return new_time, new_amp

    def pitch_shift(self, time, amp, n_steps):
        return time, np.roll(amp, int(n_steps * len(amp) / len(time)))

    def time_shift(self, time, amp, shift):
        return time - shift, amp

    def add_noise(self, amp):
        noise = np.random.normal(0, self.noise_level, len(amp))
        return amp + noise

    def apply_masks(self, amp):
        mask_ranges = []
        for _ in range(self.num_masks):
            mask_start = np.random.randint(0, len(amp))
            mask_end = mask_start + np.random.randint(int(self.mask_range[0] * len(amp)),
                                                      int(self.mask_range[1] * len(amp)))
            mask_ranges.append((mask_start, mask_end))

        for start, end in mask_ranges:
            amp[start:end] = 0

        return amp

    def apply_augmentation(self, time, amp, augmentation_prob=0.5, time_stretch_range=(0.8, 1.2),
                           pitch_shift_steps=(-4, 4), time_shift_range=(-2000, 2000)):
        augmented_time = time.copy()
        augmented_amp = amp.copy()

        if np.random.uniform() < augmentation_prob:
            rate = np.random.uniform(*time_stretch_range)
            augmented_time, augmented_amp = self.time_stretch(augmented_time, augmented_amp, rate)

        if np.random.uniform() < augmentation_prob:
            n_steps = np.random.randint(*pitch_shift_steps)
            augmented_time, augmented_amp = self.pitch_shift(augmented_time, augmented_amp, n_steps)

        if np.random.uniform() < augmentation_prob:
            augmented_amp = self.add_noise(augmented_amp)

        if np.random.uniform() < augmentation_prob:
            augmented_amp = self.apply_masks(augmented_amp)

        if np.random.uniform() < augmentation_prob:
            shift = np.random.randint(*time_shift_range)
            augmented_time, augmented_amp = self.time_shift(augmented_time, augmented_amp, shift)

        return augmented_time, augmented_amp


time = np.linspace(0, 1, num=1000)
amp = np.sin(2 * np.pi * 440 * time)

# 1. Create an instance of the AudioAugmenter class
augmenter = AudioAugmenter(noise_level=0.01, num_masks=2, mask_prob=0.5, mask_range=(0.01, 0.02))

# 2. Apply audio augmentation
augmented_time, augmented_amp \
            = augmenter.apply_augmentation(time, amp, \
                                           augmentation_prob=0.5, time_stretch_range=(0.8, 1.2),\
                                           pitch_shift_steps=(-4, 4), time_shift_range=(-2000, 2000))


# 3. Visualize the original and augmented audio waveforms
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(time, amp)
plt.title('Original Audio')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(augmented_time, augmented_amp)
plt.title('Augmented Audio')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()



# Compute the spectrograms for original and augmented audio
plt.figure(figsize=(12, 8))

# Spectrogram for original audio
plt.subplot(2, 1, 1)
plt.specgram(amp, Fs=1 / np.mean(np.diff(time)), xextent=(time[0], time[-1]))
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram - Original Audio')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

# Spectrogram for augmented audio
plt.subplot(2, 1, 2)
plt.specgram(augmented_amp, Fs=1 / np.mean(np.diff(augmented_time)), xextent=(augmented_time[0], augmented_time[-1]))
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram - Augmented Audio')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

plt.tight_layout()
plt.show()
