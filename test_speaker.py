import os
import glob
import time
import statistics as stats
import base64
from speaker_local import LocalSpeaker


TEST_DIR = "/home/lxc/projects/ai_agent/test_voicebank"
DB_DIR = "/home/lxc/projects/ai_agent/speaker_db"


def to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def infer_label_from_filename(path: str) -> str:
    fname = os.path.basename(path)
    # 取下划线前缀，例如 speaker1_a_cn_16k.wav -> speaker1
    return fname.split("_")[0]


def main():
    spk = LocalSpeaker(model_name_or_dir="campplus", db_dir=DB_DIR)
    # 如无 GPU，可改为 cpu
    spk.set_device("cuda:0")
    spk.set_resample_rate(16000)

    wav_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.wav")))
    assert wav_files, f"未找到音频：{TEST_DIR}/*.wav"

    # 选取每个说话人的第一段音频用于注册
    label_to_first = {}
    for p in wav_files:
        label = infer_label_from_filename(p)
        if label not in label_to_first:
            label_to_first[label] = p

    # 注册
    for label, path in label_to_first.items():
        b64 = to_b64(path)
        spk.register(label, b64)

    # 识别与统计
    latencies_ms = []
    confidences = []
    total = 0
    correct = 0
    details = []

    for path in wav_files:
        label = infer_label_from_filename(path)
        b64 = to_b64(path)
        t0 = time.perf_counter()
        res = spk.recognize(b64)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)
        confidences.append(res.get("confidence", 0.0) or 0.0)
        pred = res.get("name")
        ok = (pred == label)
        correct += 1 if ok else 0
        total += 1
        details.append((os.path.basename(path), label, pred, res.get("confidence", 0.0), dt))

    acc = correct / total if total else 0.0
    avg_ms = stats.fmean(latencies_ms) if latencies_ms else 0.0
    p50 = stats.median(latencies_ms) if latencies_ms else 0.0
    p95 = stats.quantiles(latencies_ms, n=20)[18] if len(latencies_ms) >= 20 else max(latencies_ms) if latencies_ms else 0.0
    avg_conf = stats.fmean(confidences) if confidences else 0.0

    print("\n=== 识别明细 ===")
    for fname, label, pred, conf, ms in details:
        print(f"{fname}\tlabel={label}\tpred={pred}\tconf={conf:.4f}\t{ms:.2f}ms")

    print("\n=== 汇总统计 ===")
    print(f"样本数: {total}")
    print(f"准确率: {acc*100:.2f}% ({correct}/{total})")
    print(f"平均耗时: {avg_ms:.2f} ms, P50: {p50:.2f} ms, P95: {p95:.2f} ms")
    print(f"平均置信度: {avg_conf:.4f}")


if __name__ == "__main__":
    main()