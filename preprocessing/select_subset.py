from datasets import load_dataset
from pathlib import Path
import json

DATASET_NAME = "nullHawk/anime-pixel-art-2"
NUM_IMAGES = 2000

OUTPUT_DIR = Path("data/anime-pixel-art-2k")
IMAGE_DIR = OUTPUT_DIR / "images"
METADATA_FILE = OUTPUT_DIR / "metadata.jsonl"


def main():
    print(f"Loading {DATASET_NAME} in streaming mode...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    count = 0

    with METADATA_FILE.open("w", encoding="utf-8") as metadata:

        for index, example in enumerate(dataset):

            tags = example["tags"]

            if isinstance(tags, list):
                tags_text = ", ".join(tags)
            else:
                tags_text = str(tags)

            tag_list = [tag.strip() for tag in tags_text.split(",")]

            # Only select images with standalone "1girl"
            if "1girl" not in tag_list:
                continue

            # Replace ONLY "1girl" -> "girl"
            tag_list = [
                "girl" if tag == "1girl" else tag
                for tag in tag_list
            ]

            caption = ", ".join(tag_list)

            filename = f"{count:04d}.png"
            image_path = IMAGE_DIR / filename

            example["image"].save(image_path)

            metadata.write(
                json.dumps(
                    {
                        "file_name": f"images/{filename}",
                        "text": caption,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            count += 1

            if count % 100 == 0:
                print(f"Saved {count}/{NUM_IMAGES}")

            # STOP immediately at 2,000
            if count == NUM_IMAGES:
                break

    print("\nDONE!")
    print(f"Saved {count} images.")
    print(f"Images:   {IMAGE_DIR}")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()