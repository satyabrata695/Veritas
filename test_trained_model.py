from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)


MODEL_DIR = Path("models/ai_detector")


def predict(image_path: str) -> None:
    image_file = Path(image_path)

    if not image_file.exists():
        raise FileNotFoundError(
            f"Image not found: {image_file}"
        )

    print("Loading model...")

    processor = AutoImageProcessor.from_pretrained(
        str(MODEL_DIR)
    )

    model = AutoModelForImageClassification.from_pretrained(
        str(MODEL_DIR)
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    image = Image.open(image_file).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    print("\n" + "=" * 55)
    print("VERITAS - TRAINED AI DETECTOR")
    print("=" * 55)
    print(f"Image: {image_file}")
    print()

    for class_id, probability in enumerate(probabilities):
        label = model.config.id2label[str(class_id)]

        print(
            f"{label:15} : "
            f"{probability.item() * 100:7.2f}%"
        )

    predicted_id = probabilities.argmax().item()

    print()
    print(
        "Prediction:",
        model.config.id2label[str(predicted_id)]
    )
    print("=" * 55)


if __name__ == "__main__":
    image_path = input(
        "Enter the full path of an image: "
    ).strip()

    predict(image_path)