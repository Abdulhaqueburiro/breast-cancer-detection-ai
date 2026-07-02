import os, sys
sys.path.append("src")
from predict import predict

images = [
    ("demo/cancer_1.png", "Cancer"),
    ("demo/cancer_2.png", "Cancer"),
    ("demo/normal_1.png", "Normal"),
    ("demo/normal_2.png", "Normal"),
]

correct = 0
for path, true in images:
    if os.path.exists(path):
        r = predict(path)
        pred = "Cancer" if "CANCER" in r["prediction"] else "Normal"
        ok = pred == true
        if ok: correct += 1
        print(f"{path}: {r['prediction']} ({r['confidence']}) - {'OK' if ok else 'WRONG'}")

print(f"Accuracy: {correct}/{len(images)}")
