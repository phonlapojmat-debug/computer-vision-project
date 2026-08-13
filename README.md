# computer-vision-project
# VisuFruit — Computer Vision Orange Grading

Experimental project using YOLOv8 Instance Segmentation to separate "orange body" from "surface defects/bruises," then calculate the defect area as a percentage of the total orange area to automate quality grading.

Built mainly for Google Colab, trained on a T4 GPU.

## Core idea

The model segments the image into classes (as trained on the dataset):
- Class 2 = orange fruit body (used as the base area)
- Class 0 and 1 = defect types (e.g. bruises, mold)

Formula:

```
Damage Ratio (%) = (defect area / orange area) × 100
```

## Grading criteria

| Grade | Condition | Action |
|---|---|---|
| A | damage < 2% | full-price retail (supermarkets) |
| B | damage 2–10% | juice processing plants / discount |
| C | damage > 10% | reject, pull from inventory immediately |

## Notebook structure

The notebook goes step by step from loading the dataset to running evaluation:

1. Install `ultralytics` and `roboflow`
2. Download dataset from Roboflow (you need your own API key)
3. Train YOLOv8n-seg on the dataset (10 epochs, image size 320)
4. Check sample image dimensions in the dataset
5. Run inference on the test set and view segmentation output
6. Calculate defect ratio and grade for a single image
7. Evaluate the model with `model.val()` (box and mask mAP50)
8. Run inference + grading over the whole test set in a loop

After that, the notebook walks through two versions of the defect-counting logic:

- **Version 1**: only counts class 1 as defect (`elif class_id == 1`) — the problem is that if the model also detects class 0 (e.g. bruises), those pixels get ignored, which understates the damage %.
- **Version 2**: sums up every class that isn't class 2 (orange body) as defect (`else: defect_pixels += ...`) — more complete.

There's also a bug fix: when the model detects defects but fails to detect the orange body (orange_pixels = 0), the original code just did `continue` and skipped the image — risking a division by zero and leaving gaps in the report. The fix adds a fallback: treat it as 100% damage and grade C automatically.

## How to run

1. Open this notebook in Google Colab (GPU recommended)
2. Run the install/Roboflow cells — you need your own Roboflow API key (sign up at roboflow.com and swap it in for `YOUR_ROBOFLOW_API_KEY`)
3. Run the training cell (takes a while even at 10 epochs)
4. Run the inference/grading cells in order

## What I got out of building this

- Got real hands-on experience with instance segmentation (YOLOv8-seg), not just plain object detection — it made the difference between bounding boxes and pixel-level masks concrete, and why pixel-level output is what you actually need for an "area ratio" calculation like this.
- Ran into a real bug from my own logic (the orange_pixels = 0 case), which drove home how much edge-case testing matters — the model itself can be working fine while the surrounding code quietly breaks.
- Saw directly how much the choice of counting logic (version 1 vs 2) affects the final result — picking the narrower one understates defects, which in a real deployment could let genuinely damaged fruit slip through as Grade A.
- Practiced the full workflow with a public Roboflow dataset — download, train, evaluate — end to end on Colab.

## Where this could go next

- Train for longer than 10 epochs and tune hyperparameters to get accuracy (mAP) up to something trustworthy enough for real use.
- Expand/adjust the dataset to actually match the lighting, camera angles, and orange varieties this would be used on, instead of relying on the public dataset as-is.
- Tie the grading thresholds (2%, 10%) to real business quality standards instead of arbitrary numbers.
- Remove the Colab-only hardcoded paths (`/content/...`) so it's portable to other environments.
- Move beyond the notebook into an actual deployed system — e.g. an API, or real-time inference on a camera over a sorting conveyor.
- Add stats/dashboard tracking to see quality trends across batches over time, instead of grading one image at a time in isolation.
