# Watermark Detector - Detectron2
### Generate dataset
Ensure you have photos and watermarks split into `train`, `test` and `val` under `data`. 
The folder structure should be as follows.
```bash
data/
├── test
│   ├── photos
│   └── watermarks
├── train
│   ├── photos
│   └── watermarks
└── val
    ├── photos
    └── watermarks
```
Run `python3 generate_dataset.py` to generate input and binary images for training. 
Here is the folder structure after dataset is generated.
```bash 
data/
├── test
│   ├── input
│   ├── mask_watermark
│   ├── mask_word
│   ├── photos
│   └── watermarks
├── train
│   ├── input
│   ├── mask_watermark
│   ├── mask_word
│   ├── photos
│   └── watermarks
└── val
    ├── input
    ├── mask_watermark
    ├── mask_word
    ├── photos
    └── watermarks
```

### Initialising Docker
Build docker image:
```bash
docker build -t watermark-detector:v0 .
```
Run docker container:
```bash
bash run_docker_gpu.bash
```

### Training
Run `train_watermarks.py train` to begin training. Detectron2 will create an `output` folder for storing logs.
At the end of training, Detectron2 will return AP scores on test set in the form of `OrderDict`.

### Inference
Run `train_watermarks.py test` for inference. 
Inference will be run on the test set in `data` as well as any other datasets in `dataset`.
Results will be stored in the `output` folder as during training.

### Bounding Box Benchmark Results:
<table>
<thead>
  <tr>
    <th>backbone</th>
    <th>AP</th>
    <th>AP50</th>
    <th>AP75</th>
    <th>APs</th>
    <th>APm</th>
    <th>APl</th>
    <th>AP-watermark</th>
    <th>AP-text</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>R_50_FPN_3x</td>
    <td>45.5</td>
    <td>68.0</td>
    <td>51.3</td>
    <td>38.5</td>
    <td>63.5</td>
    <td>70.3</td>
    <td>31.5</td>
    <td>59.4</td>
  </tr>
  <tr>
    <td>R_50_C4_3x</td>
    <td>46.6</td>
    <td>66.6</td>
    <td>51.6</td>
    <td>38.5</td>
    <td> 68.0</td>
    <td>72.9</td>
    <td>33.4</td>
    <td>59.8</td>
  </tr>
  <tr>
    <td>R_50_DC5_3x</td>
    <td>45.1</td>
    <td>66.5</td>
    <td>50.72</td>
    <td>36.7</td>
    <td>66.1</td>
    <td>58.9</td>
    <td>32.3</td>
    <td>57.8</td>
  </tr>
  <tr>
    <td>R_101_DC5_3x</td>
    <td>45.6</td>
    <td>67.2</td>
    <td>51.5</td>
    <td>37.1</td>
    <td>68.0</td>
    <td>65.0</td>
    <td>33.3</td>
    <td>57.9</td>
  </tr>
  <tr>
    <td>R_101_FPN_3x</td>
    <td>47.06</td>
    <td>68.8</td>
    <td>52.5</td>
    <td>39.0</td>
    <td>66.8</td>
    <td>80.0</td>
    <td>33.7</td>
    <td>60.3</td>
  </tr>
  <tr>
    <td>X_101_32x8d_FPN_3x</td>
    <td> 51.7</td>
    <td>70.9</td>
    <td>56.9</td>
    <td>43.5</td>
    <td>70.2</td>
    <td>14.8</td>
    <td>34.9</td>
    <td>68.5</td>
  </tr>
</tbody>
</table>

### Segmentation Benchmark Results:
<table>
<thead>
  <tr>
    <th>Backbone</th>
    <th>AP</th>
    <th>AP50</th>
    <th>AP75</th>
    <th>APs</th>
    <th>APm</th>
    <th>APl</th>
    <th>AP-watermark</th>
    <th>AP-text</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>R_50_FPN_3x</td>
    <td>38.9</td>
    <td>65.9</td>
    <td>42.5</td>
    <td>31.3</td>
    <td>53.2</td>
    <td>90.0</td>
    <td>26.9</td>
    <td>50.9</td>
  </tr>
  <tr>
    <td>R_50_C4_3x</td>
    <td>33.9</td>
    <td> 63.2</td>
    <td>31.5</td>
    <td> 26.0</td>
    <td>50.0</td>
    <td>90.0</td>
    <td>25.9</td>
    <td>42.0</td>
  </tr>
  <tr>
    <td>R_50_DC5_3x</td>
    <td>35.6</td>
    <td>63.7</td>
    <td>35.5</td>
    <td>26.9</td>
    <td>54.1</td>
    <td>60.0</td>
    <td>26.6</td>
    <td>44.7</td>
  </tr>
  <tr>
    <td>R_101_DC5_3x</td>
    <td>35.6</td>
    <td>65.0</td>
    <td>34.5</td>
    <td>26.5</td>
    <td>55.5</td>
    <td>46.6</td>
    <td>27.9</td>
    <td>43.3</td>
  </tr>
  <tr>
    <td>R_101_FPN_3x</td>
    <td>39.2</td>
    <td> 66.8</td>
    <td>41.9</td>
    <td>31.4</td>
    <td>54.5</td>
    <td>90.0</td>
    <td>28.1</td>
    <td>50.4</td>
  </tr>
  <tr>
    <td>X_101_32x8d_FPN_3x</td>
    <td>43.3</td>
    <td> 68.8</td>
    <td>49.2</td>
    <td>35.7</td>
    <td>55.8</td>
    <td> 65.0</td>
    <td>29.0</td>
    <td>57.5</td>
  </tr>
</tbody>
</table>
# watermark-detector
