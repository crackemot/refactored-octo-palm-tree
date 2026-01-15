import numpy as np
import librosa
import os
import json
import warnings

warnings.filterwarnings("ignore")


class STSDetector:
    def __init__(self, config_path: str = "sts_config.json"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

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

    def analyze(self, audio_path: str) -> dict:
        if not os.path.exists(audio_path):
            return {'error': 'File not found'}

        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            y = y / (np.max(np.abs(y)) + 1e-9)
            features = self.extract_features(y, sr)

            score = 0.0
            breakdown = {}

            for feat_name, weight in self.config['weights'].items():
                if feat_name not in features:
                    continue
                value = features[feat_name]
                threshold = self.config['thresholds'][feat_name]
                direction = self.config['directions'][feat_name]

                triggered = (
                    (direction == 'greater' and value > threshold) or
                    (direction == 'less' and value < threshold)
                )

                if triggered:
                    score += weight
                    breakdown[feat_name] = {
                        'value': value,
                        'threshold': threshold,
                        'direction': direction,
                        'weight': weight
                    }

            score = min(score, 1.0)
            is_fake = score > self.config['vc_threshold']

            return {
                'file': os.path.basename(audio_path),
                'score': score,
                'is_fake': is_fake,
                'verdict': 'VC/STS' if is_fake else 'LIVE',
                'features': features,
                'breakdown': breakdown
            }

        except Exception as e:
            return {'error': str(e)}


import argparse

def main():
    parser = argparse.ArgumentParser(
        description="VC/STS Detector"
    )
    parser.add_argument(
        "files", nargs="+", metavar="FILE", help="Один или несколько аудиофайлов (.wav, .ogg)"
    )
    parser.add_argument(
        "-c", "--config", default="sts_config.json",
        help="Путь к конфигурации (по умолчанию: sts_config.json)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Показать значения всех признаков (не только сработавших)"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Только вердикт + балл (для скриптов/пайплайнов)"
    )

    args = parser.parse_args()

    detector = STSDetector(args.config)

    for audio_path in args.files:
        if not os.path.exists(audio_path):
            if not args.quiet:
                print(f" {audio_path}: файл не найден")
            continue

        result = detector.analyze(audio_path)

        if 'error' in result:
            if not args.quiet:
                print(f" {audio_path}: ошибка — {result['error']}")
            continue

        if args.quiet:
            print(f"{result['file']}\t{result['verdict']}\t{result['score']:.4f}")
            continue

        color = "\033[91m" if result['is_fake'] else "\033[92m"
        reset = "\033[0m"
        print(f"\n{result['file']}:")
        print(f"  Вердикт: {color}{result['verdict']}{reset}")
        print(f"  Сводный балл: {result['score']:.4f}")

        if args.verbose:
            print("  Все признаки:")
            for feat, val in result['features'].items():
                thr = detector.config['thresholds'][feat]
                dir_ = detector.config['directions'][feat]
                sym = ">" if dir_ == 'greater' else "<"
                status = "✓" if (
                    (dir_ == 'greater' and val > thr) or
                    (dir_ == 'less' and val < thr)
                ) else "–"
                print(f"    {feat}: {val:.4f} {sym} {thr:.4f} {status}")

        elif result['breakdown']:
            print("  Сработавшие признаки:")
            for feat, data in result['breakdown'].items():
                sym = ">" if data['direction'] == 'greater' else "<"
                print(f"    {feat}: {data['value']:.4f} {sym} {data['threshold']:.4f} "
                      f"(вес {data['weight']:.3f})")

if __name__ == "__main__":
    main()