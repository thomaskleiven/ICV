# Image-Processing-Computer-Vision

An algorithm for Content Based Image Retrieval (CBIR) with Wavelet features. The system takes in a query image and tries 
to find the most 'semantically similar' images in a database of images. The CBIR system uses Discrete Wavelet 
Transform to generate a core image feature representation in order to compare images.

First, build the database and store all the obtained image feature vectors in a KD-tree:

```python
python3 buildDatabase.py ./path_to_images
```
Subsequently, query the obtained database by running:

```bash
python3 query.py ./path_to_image
```

### Example
```python
python3 buildDatabase.py ./images
```

```bash
python3 query.py ./query-horse.jpg
```
