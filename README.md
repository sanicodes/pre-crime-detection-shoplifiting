# Shoplifting detection from CCTV video

Classifies surveillance videos as shoplifting or normal, using per-frame features from a pre-trained ResNet-50 and an LSTM over time. This started as a thesis project ("pre-crime detection of shoplifting"); the notebook here is a re-run with the bugs in the original code fixed.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sanicodes/pre-crime-detection-shoplifiting/blob/main/shoplifting_detection.ipynb)

| Path | |
|---|---|
| `shoplifting_detection.ipynb` | data download, feature extraction, training and evaluation |
| `thesis_original/` | the 12 original thesis notebooks |

## Method

```
CCTV video ──► frames (sampled by timestamp, RGB, 224×224)
           ──► ResNet-50 V2, ImageNet weights, frozen ──► 2048 features per frame (computed once)
           ──► sequence in time order, padded and masked
           ──► Dense(128) ──► LSTM(128) ──► P(shoplifting)
```

Evaluated with stratified 5-fold cross-validation, repeated over several seeds, and compared against:

- logistic regression on video length, a single frame, and the mean of all frames;
- the same LSTM with each video's frames shuffled;
- the original thesis pipeline, re-created with its bugs;
- a pre-theft test, on footage from before the annotated start of each theft.

## Data

[UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/): the 50 Shoplifting videos and the 50 videos in `Normal_Videos_for_Event_Recognition`, plus the temporal annotations for the pre-theft test. The notebook downloads them from a [Hugging Face mirror](https://huggingface.co/datasets/jinmang2/ucf_crime) and fetches only the needed files (about 3.7 GB of the 7.7 GB in the two zips). UCF-Crime is released for research use.

## Running

**Colab:** open the notebook from the badge, choose a T4 GPU runtime and run all. Features and results are saved to `MyDrive/shoplifting-rerun`, so running again after a disconnect resumes where it stopped. `SMOKE_TEST = True` in the first code cell does a quick check run on 12 short videos.

**Locally:** `pip install -r requirements.txt` and run the notebook with Jupyter. Feature extraction for the full dataset takes hours without a GPU.

## References

- Sultani, Chen & Shah, [Real-world Anomaly Detection in Surveillance Videos](https://arxiv.org/abs/1801.04264), CVPR 2018
- He, Zhang, Ren & Sun, [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027), ECCV 2016
