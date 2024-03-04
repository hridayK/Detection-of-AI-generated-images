# Detection-of-AI-generated-images

Reference paper - [link](https://arxiv.org/abs/2311.12397)

Datasets used: 
- [ArtiFact: Real and Fake Image Dataset](https://www.kaggle.com/datasets/ravidussilva/real-ai-art)
- [AI recognition dataset](https://www.kaggle.com/datasets/superpotato9/dalle-recognition-dataset/data)

**Note** - All the mentioned datasets are partially used, training isn't done on all of their data.

Blog - [link](https://medium.com/p/fc2024e3e716)

An attempt to detect AI generated images in a generalized manner.

# Logs:
**Update: 17/02/2024**
- Removed noise adding filters.
- experimented with usage of *pixel_fluctuation_ratio* as a feature to learn if image is ai generated or not. (refer to ```pixel_fluctuation.ipynb```).

**Update: 18/02/2024**
- The standard matrix rotation function of ```scipy.ndimage.rotate``` uses affine transformation to rotate a matrix while there are more ways to rotate a matrix, attempt to explore methods to avoid noisy outputs after applying filters.

- **Result** - The way of rotation wasn't a problem but the way to merge filters was causing noisy outputs.

**Update: 20/02/2024**
- Implement data pipeline.
- Trained for till: ```loss: 0.1729 - binary_accuracy: 0.924```.
- saved model as ```classifier.h5```.

**Update: 21/02/2024**
- implement testing notebook.


**Update: 04/03/2024**
- optimize for matrix rotation while applying filters.
