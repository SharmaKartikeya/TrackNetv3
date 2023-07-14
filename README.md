## Kartikeya's Documentation

The gaussian distribution heatmaps are not used. Instead, a simple ellipse like structure is used to represent the ball in the heatmaps. The height and the width are scaled proportionally.

---

The structure of the data has been changed. Since we have yolo annotations, we have made use of the width and height information of the ball in every frame. The `.csv` files have the following columns
  
  ```
  frame_num, x, y, w, h, visible
  ```

  The x and y are the normalized x and y coordinates of the center of the ball. The normalization is done w.r.t. the size of the frame, for example (512, 1024) where 512 is height and 1024 is width

  The w and h are the width and height of the bounding box extracted from the yolo annotations that will be the lengths of the axes of the ellipses used in the heatmaps

  Similar to TrackNetv2, visible will be 1 if the ball is visible (clear or blurred) and 0 if it is occluded or not visible

---

EfficientNet-B1 backbone is used from `pytorch-segmentation-models` for extracted features from the input.

---

In case the `sequence_length = 3` and `grayscale = False`, the input will be (9, 512, 1024) and the output heatmaps will be (3, 512, 1024). One grayscale heatmap per sequence frame.

Example input frame of sequence_length = 1:

<img src="assets/example_input_frame.PNG" width='512'>

Corresponding heatmap:

<img src="assets/example_heatmap.PNG" width='512'>

---

## train.py Parameters cheatsheet
| Argument name      | Type  | Default value | Description |
|--------------------|-------|---------------|-------------|
|weights                |str    |None           |Path to initial weights the model should be loaded with. If not specified, the model will be initialized with random weights.|
|checkpoint             |str    |None           |Path to a checkpoint, chekpoint differs from weights by to including information about current loss, epoch and optimizer state.|
|batch_size             |int    |2              |Batch size of the training dataset.|
|val_batch_size         |int    |1              |Batch size of the validation dataset.|
|shuffle                |bool   |True           |Should the dataset be shuffled before training?|
|epochs                 |int    |10             |Number of epochs.|
|train_size             |float  |0.8            |Training dataset size.|
|lr                     |float  |0.01           |Learning rate.|
|momentum               |float  |0.9            |Momentum.|
|dropout                |float  |0.0            |Dropout rate. If equals to 0.0, no dropout is used.|
|dataset                |str    |'dataset/'     |Path to dataset.|
|device                 |str    |'cpu'          |Device to use (cpu, cuda, mps).|
|type                   |str    |'auto'         |Type of dataset to create (auto, image, video). If auto, the dataset type will be inferred from the dataset directory, defaulting to image.|
|save_period            |int    |10             |Save checkpoint every x epochs (disabled if <1).|
|save_weights_only      |bool   |False          |Save only weights, not the whole checkpoint|
|save_path              |str    |'weights/'     |Path to save checkpoints at.|
|no_shuffle             | -     | -             |Don't shuffle the training dataset.|
|tensorboard            | -     | -             |Use tensorboard to log training progress.')|
|one_output_frame       | -     | -             |Demand only one output frame instead of three.')|
|no_save_output_examples| -     | -             |Don't save output examples to results folder.|
|grayscale              | -     | -             |Use grayscale images instead of RGB.')|
|single_batch_overfit   | -     | -             |Overfit the model on a single batch.')|

Arguments without type or default value are used without an additional value, e.x.

Credits: https://github.com/mareksubocz/TrackNet/tree/main