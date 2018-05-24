# Image-Processing-Computer-Vision

An algorithm for Content Based Image Retrieval (CBIR) with Wavelet features. The system take in a query image and try 
to find the most 'semantically similar' images in a database of images. The CBIR system uses Discrete Wavelet 
Transform to generate a core image feature representation in order to compare images.

First, build the database and store all the obtained feature vectors in a KD-tree:

```python
python3 buildDatabase.py ./path_to_images
```
This will build and store all the feature vectors of the images in your database and store it in a KD-tree.

Subsequently, query the obtained database by running:

```bash
python3 query.py ./path_to_image
```
