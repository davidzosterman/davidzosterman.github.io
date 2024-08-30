---
title: "Selected Projects"
permalink: /projects/
author_profile: true
---

<a id="googlemapsrestar"></a>

## Google Maps Re-Star

Employing Natural Language Processing (NLP) to re-rate restaurants using Google Maps written reviews.

<img src="images/visualize_documents-header.png" alt="Nice Header Plot" width="300"/>

### Data Scraping and Pre-processing

The data consists of 645 customer reviews for the NYC restaurant Bangia, scraped from Google Maps using the Google Maps `Places API`.
Only the 344 reviews that have written comments were used.

The dataframe contains 3 columns:
- `Id`: Effectively the index of the row, used for merging dataframes during processing and analysis. Int.
- `rating`: 1 to 5 star rating of the restaurant. Increments of 1 star Int.
- `text`: The text of the review. String.

### Model - RoBERTa by Huggingface

We use a pre-trained RoBERTa model from Huggingface (https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment).
The model performs sentiment analysis on text and assigns positive, negative, and neutral (`roberta_pos`, `roberta_neg`, and `roberta_neu` respectively) scores to each of the reviews.

The data all exists in the plane `roberta_pos`+`roberta_neg`+`roberta_neu`=1, so we create two new RQs from convenient basis vectors in this plane:
- `pos_neg`=`roberta_pos`-`roberta_neg`
- `pos_neg_2neu`=`roberta_pos`+`roberta_neg`-2*`roberta_neu`

We then find a best fit hyperbola to the data, find the nearest point on the curve to each data point, and assign that data point a new star rating equal to the fraction of its distance along the curve times 4.0 plus 1.0.

<img src="images/visualize_documents-3-D.png" alt="Re-Star Bangia Reviews" width="600"/>

### Processed Data
The final dataframe contains the 3 columns of the original dataframe, plus 6 new columns:
- `roberta_pos`: RoBERTa sentiment analysis positive score. `roberta_pos`+`roberta_neg`+`roberta_neu`=1. Float.
- `roberta_neg`: RoBERTa sentiment analysis negative score. `roberta_pos`+`roberta_neg`+`roberta_neu`=1. Float.
- `roberta_neu`: RoBERTa sentiment analysis neutral score. `roberta_pos`+`roberta_neg`+`roberta_neu`=1. Float.
- `new_stars`: New 1 to 5 star rating of the restaurant from this review. Increments of 0.1 stars. Minimum of 1.0 stars. Int.
- `pos_neg`: First orthogonal basis vector in the `roberta_pos`+`roberta_neg`+`roberta_neu`=1 plane.`pos_neg`=`roberta_pos`-`roberta_neg`. Float.
- `pos_neg_2neu`: Second orthogonal basis vector in the `roberta_pos`+`roberta_neg`+`roberta_neu`=1 plane. `pos_neg_2neu`=`roberta_pos`+`roberta_neg`-2*`roberta_neu`. Float.
