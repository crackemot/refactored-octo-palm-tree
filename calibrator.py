import numpy as np
import librosa
import os
import json
import glob
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")


class STSCalibrator:
    def __init__(self, config_path: str = "sts_config.json"):
        self.config_path = config_path
        self.real_folder = "real"
        self.fake_folder = "fake"

    def extract_features(self, y: np.ndarray, sr: int) -> dict[str, float]:
        features = {}

        flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness'] = float(np.mean(flatness))

        try:
            harmonic = librosa.effects.harmonic(y, margin=8)
            percussive = librosa.effects.percussive(y, margin=8)
            h_energy = float(np.sum(harmonic ** 2))
            p_energy = float(np.sum(percussive ** 2))
            features['harmonic_ratio'] = h_energy / (h_energy + p_energy + 1e-9)
        except Exception:
            features['harmonic_ratio'] = 0.5

        try:
            f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=500, sr=sr)
            valid_f0 = f0[voiced & ~np.isnan(f0)]
            if len(valid_f0) > 10:
                features['pitch_std'] = float(np.std(valid_f0))
            else:
                features['pitch_std'] = 50.0
        except Exception:
            features['pitch_std'] = 50.0

        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_variation'] = float(np.std(zcr))

        return features

    def calibrate_fixed_weights(self) -> dict:
        real_files = glob.glob(os.path.join(self.real_folder, "*.wav")) + \
                     glob.glob(os.path.join(self.real_folder, "*.ogg"))
        fake_files = glob.glob(os.path.join(self.fake_folder, "*.wav")) + \
                     glob.glob(os.path.join(self.fake_folder, "*.ogg"))

        real_features = []
        for file in real_files:
            try:
                y, sr = librosa.load(file, sr=None, mono=True)
                y = y / (np.max(np.abs(y)) + 1e-9)
                features = self.extract_features(y, sr)
                real_features.append(features)
            except Exception:
                continue

        fake_features = []
        for file in fake_files:
            try:
                y, sr = librosa.load(file, sr=None, mono=True)
                y = y / (np.max(np.abs(y)) + 1e-9)
                features = self.extract_features(y, sr)
                fake_features.append(features)
            except Exception:
                continue

        if not real_features or not fake_features:
            raise ValueError("Недостаточно данных")

        avg_real = {
            key: np.mean([f[key] for f in real_features])
            for key in real_features[0].keys()
        }
        avg_fake = {
            key: np.mean([f[key] for f in fake_features])
            for key in fake_features[0].keys()
        }

        thresholds = {
            key: (avg_real[key] + avg_fake[key]) / 2
            for key in avg_real.keys()
        }

        directions = {
            key: 'greater' if avg_fake[key] > avg_real[key] else 'less'
            for key in avg_real.keys()
        }

        weights = {
            'spectral_flatness': 0.40,
            'harmonic_ratio': 0.35,
            'pitch_std': 0.15,
            'zcr_variation': 0.10,
        }

        config = {
            'thresholds': thresholds,
            'directions': directions,
            'weights': weights,
            'avg_real': avg_real,
            'avg_fake': avg_fake,
            'vc_threshold': 0.55,
            'calibration_info': {
                'real_files_count': len(real_features),
                'fake_files_count': len(fake_features),
                'timestamp': datetime.now().isoformat()
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        return config

    def print_report(self, config: dict):
        print("\n" + "=" * 80)
        print("КАЛИБРОВКА С ФИКСИРОВАННЫМИ ВЕСАМИ")
        print("=" * 80)
        print(f"\nСтатистика:")
        print(f"  REAL файлов: {config['calibration_info']['real_files_count']}")
        print(f"  FAKE файлов: {config['calibration_info']['fake_files_count']}")
        print(f"\nПризнаки и пороги:")
        print(f"{'Признак':<25} {'REAL':<10} {'FAKE':<10} {'Порог':<10} {'Вес':<8}")
        print("-" * 80)
        for key in config['weights'].keys():
            real_val = config['avg_real'][key]
            fake_val = config['avg_fake'][key]
            threshold = config['thresholds'][key]
            weight = config['weights'][key]
            direction = ">" if config['directions'][key] == 'greater' else "<"
            print(f"{key:<25} {real_val:<10.4f} {fake_val:<10.4f} "
                  f"{threshold:<10.4f} {weight:<8.3f} ({direction} FAKE)")
        print(f"\nПорог классификации: {config['vc_threshold']}")
        print(f"Конфигурация: {self.config_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("VC/STS Detector Calibrator")
    print("=" * 80)

    calibrator = STSCalibrator("sts_config.json")
    try:
        config = calibrator.calibrate_fixed_weights()
        calibrator.print_report(config)
        print("\n Калибровка завершена!")
    except Exception as e:
        print(f"\n Ошибка: {e}")