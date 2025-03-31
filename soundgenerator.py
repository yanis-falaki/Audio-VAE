from preprocess import MinMaxNormalizer
import librosa
import soundfile as sf
import os


class SoundGenerator:
    """SoundGenerator is responsible for generating audio from
    spectrograms."""

    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normalizer = MinMaxNormalizer(0, 1)

    def generate(self, spectrograms, min_max_values):
        generated_spectrograms, latent_representations = self.vae.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals, latent_representations
    
    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):

            # reshape log spectrogram
            log_spectrogram = spectrogram.squeeze(0)

            # apply denormalization
            denorm_log_spec = self._min_max_normalizer.denormalize(
                log_spectrogram, min_max_value["min"], min_max_value["max"])
            
            # log spectrogram -> spectrogram
            spec = librosa.db_to_amplitude(denorm_log_spec)

            # apply Griffin-Lim algorithm
            signal = librosa.istft(spec, hop_length=self.hop_length)

            # append signal to "signals"
            signals.append(signal)

        return signals
    
    def save_signals(signals, save_dir, sample_rate=22050):
        for i, signal in enumerate(signals):
            save_path = os.path.join(save_dir, str(i) + ".wav")
            sf.write(save_path, signal, sample_rate)