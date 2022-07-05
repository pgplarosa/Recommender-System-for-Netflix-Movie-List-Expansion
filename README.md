# Oldies but Goodies: Recommender System for Netflix Movie List Expansion

Big Data and Cloud Computing Mini Project 1 by Marvee dela Resma and Patrick Guillano La Rosa - February 12, 2022. The code, analysis, and the full report are included in the [Technical Report](https://github.com/pgplarosa/Recommender-System-for-Netflix-Movie-List-Expansion/blob/main/md/main.md). If you have any questions regarding this study, please send me a message via  [LinkedIn](https://www.linkedin.com/in/patricklarosa/).

<img src="outputs/banner.png" width=100% />

## Executive Summary

<p align="justify">The business value of recommender systems in digitally reliant industries is undeniable. User preference is the most important aspect of any product-oriented business so providing the best predicti.on for the next purchase is a big advantage. Aside from this, recommender systems could also be used to increase the variety of products currently available.</p>

<p align="justify">In this technical paper, two separate datasets (Movielens & Netflix) containing user rating data for movies and shows, were utilized to expand the show line-up of Netflix. Movielens, a movie recommendation site has an extensive record of movie data with corresponding ratings from its community members. The goal is to identify some movies from this selection which could be added to the current Netflix show line up. After identifying common users from both datasets, the chosen Netflix users were added to the Movielens utility matrix to generate predicted ratings for all movies and in turn provide a list of recommendations. Two latent-based methods Singular Value Decomposition (SVD) and Non-negative Matrix Factorization (NMF) were used for recommendation generation. The SVD model produced 4,717 unique movies, meanwhile using NMF produced 9,471 movies to recommend to Netflix. Using error metrics such as RMSE, MSE, and MAE, these models were then evaluated where SVD was seen as the better performing model between the two. The top recommendation item sets were then analyzed and chosen using frequent itemset mining methods by applying minimum support to generate a final list of shows recommended for inclusion in Netflix’s show line up.</p>

<p align="justify">We recommend future researchers to expand the scope of movie rating statistics and update rating data to the latest available one. We also suggest trying neighborhood based collaborative filtering (CF) methods and to introduce CF models in combination with content-based recommender systems.</p>
