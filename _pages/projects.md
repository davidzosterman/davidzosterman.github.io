---
title: "Selected Projects"
permalink: /projects/
author_profile: true
---

<a id="googlemapsrestar"></a>

# Google Maps Re-Star
---------------------
Employing Natural Language Processing (NLP) to re-rate restaurants using Google Maps written reviews.

<img src="../images/visualize_documents-header.png" alt="Nice Header Plot" width="300"/>

## Data Scraping and Pre-processing

The data consists of 645 customer reviews for the NYC restaurant Bangia, scraped from Google Maps using the Google Maps `Places API`.
Only the 344 reviews that have written comments were used.

The dataframe contains 3 columns:
- `Id`: Effectively the index of the row, used for merging dataframes during processing and analysis. Int.
- `rating`: 1 to 5 star rating of the restaurant. Increments of 1 star Int.
- `text`: The text of the review. String.

## Model - RoBERTa by Huggingface

We use a pre-trained RoBERTa model from Huggingface (https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment).
The model performs sentiment analysis on text and assigns positive, negative, and neutral (`roberta_pos`, `roberta_neg`, and `roberta_neu` respectively) scores to each of the reviews.

The data all exists in the plane `roberta_pos`+`roberta_neg`+`roberta_neu`=1, so we create two new RQs from convenient basis vectors in this plane:
- `pos_neg`=`roberta_pos`-`roberta_neg`
- `pos_neg_2neu`=`roberta_pos`+`roberta_neg`-2*`roberta_neu`

We then find a best fit hyperbola to the data, find the nearest point on the curve to each data point, and assign that data point a new star rating equal to the fraction of its distance along the curve times 4.0 plus 1.0.

<img src="../images/visualize_documents-3-D.png" alt="Re-Star Bangia Reviews" width="600"/>

## Processed Data

The final dataframe contains the 3 columns of the original dataframe, plus 6 new columns:
- `roberta_pos`: RoBERTa sentiment analysis positive score. `roberta_pos`+`roberta_neg`+`roberta_neu`=1. Float.
- `roberta_neg`: RoBERTa sentiment analysis negative score. `roberta_pos`+`roberta_neg`+`roberta_neu`=1. Float.
- `roberta_neu`: RoBERTa sentiment analysis neutral score. `roberta_pos`+`roberta_neg`+`roberta_neu`=1. Float.
- `new_stars`: New 1 to 5 star rating of the restaurant from this review. Increments of 0.1 stars. Minimum of 1.0 stars. Int.
- `pos_neg`: First orthogonal basis vector in the `roberta_pos`+`roberta_neg`+`roberta_neu`=1 plane.`pos_neg`=`roberta_pos`-`roberta_neg`. Float.
- `pos_neg_2neu`: Second orthogonal basis vector in the `roberta_pos`+`roberta_neg`+`roberta_neu`=1 plane. `pos_neg_2neu`=`roberta_pos`+`roberta_neg`-2*`roberta_neu`. Float.


<a id="bodybuildingjudge"></a>

# Finding limits of deep learning with bodybuilder image comparison
--------------------
Erdos Institute Deep Learning Bootcamp project

Fall 2024

- Luke Corwin
- David Osterman
- Andrei Prokhorov

--------------------

Creating a two-input convolutional neural network to decide the winner between two bodybuilder photos.

<img src="./other_images/cover_image.png" alt="Nice Header Plot" width="300"/> <img src="./other_images/cover_image_2.jpg" alt="Nice Header Plot" width="300"/>

## Data Description

Data consisted of bodybuilder competitor photos from the [National Physique Committee website](https://contests.npcnewsonline.com/contests/2024). Photos were downloaded using Beautiful Boup in Google Colab. To reduce training time and needed model complexity, we focused on competitions from the `year` 2024 in the `division` 'Men’s Physique'. Within each competition, each contestant has a variable number of photos. We manually selected one front photo from each contestant. Each competition also listed the final `rank` of the competitors from 1 up to 16. If more than 16 competitors were present, all after the first fifteen were tied at 16th place.

The purpose of our model was to compare an image from each of two competitors and decide a winner. Our dataset input `x` consisted of pairs of two photos, the front-facing photos from two different competitors, and our dataset `y` output consisted of an integer 0 (first competitor wins) or 1 (second competitor wins). Within each competition, we added a datapoint for each pair of competitors, using the rankings within the competition to determine if the winner should be 0 or 1. Two important nuances were:
- Each pair of competitors added *two* datapoints for each competitor pair, with the order of the competitors as competitor 0 or 1 reversed. This was to prevent the network's two image recognition parts from learning features of different competitors.
- If two competitors had the same ranking (eg. two competitors with ranking 16 or one competitor paired against himself) then the datapoint was thrown out.

We divided the data into training, validation, and test sets, with a split of 65-18-17. The training set consisted of 11,776 image pairs with one-hot-coded labels. To keep the integrity of our validation and test sets, we kept competitions intact and ensured no competitor appeared in more than one of the three sets.

The data was formatted into an instance of a custom dataset we created, `CustomDataset(Dataset)`, a subclass of `torch.utils.data.Dataset`. The photos, originally 600x900, were transformed to 128x128 using `torchvision.transforms.Resize()`.

## Models

We employed two different models, an original two-input convolutional neural network and a ResNet50-based network.

We first created an original model called `TwoInputNet(nn.Module)`, a subclass of `torch.nn.Module`. The architecture is the following:
- A sequential set of layers to learn features of the first input image. There were three convolution layers, each followed by a ReLU activation function and a 2D MaxPool. 
- A sequential set of layers to learn features of the second input image. It is identical to that of the first input image
- A sequential set of layers to combine the outputs from the two sets of convolutional layers. There was a linear layer, followed by a ReLU activation, followed by a linear layer with an output of size 1, and finally a sigmoid activation layer.

An output less than 0.5 was rounded down to 0 to indicate the first input image was the winner. An output greater than 0.5 was rounded up to 1 to indicate the second input image was the winner.

The ResNet50-based model architecture was identical to that of our original model, except that the two sequential image-recognition parts were each replaced with ResNet50.

Both networsk were trained for 20 epochs with 108 workers and a batch size of 256 on an A100 GPU in Google Colab. 

## Results

We measured the loss and accuracy for the training and validation set for each epoch of training. Despite varying multiple hyperparameters, including learning rate and number of epochs, we were unsuccessful in validating either of the models. In every case, we found that either the model did not learn (training and validation loss functions remained flat)

<img src="./other_images/orig_net_not_training.png" alt="Original Net Not Training" width="600"/>

or overfit (training loss function decreased while validation loss function increased.

<img src="./other_images/resnet_overfit.png" alt="ResNet50 Overfit" width="600"/>

Potential improvements include  investigating the behavior of our models carefully and collecting more data from the NPC website to add more photo pairs. It may also be useful to train a separate network to facilitate the photo selection process.
