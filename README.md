## Example Arguments

## Model Training
``` python train.py --style_path "./imgs/style/" --content_path "/path/to/content/dataset/" ```

## Model Evaluation
``` python evaluate.py --content_path "./imgs/gentlecat.png" --style_index -1 --output_path stylized.png ```

### For training
    - Download the content dataset first
    - Content Images dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k/
    - Style Images are already Provided in imgs/styles

### For virtual env/conda env setup use python 3.9 and install requirements.txt, later in if any requirement isn't found install manually