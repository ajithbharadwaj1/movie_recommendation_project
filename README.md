# Movie Recommendation System
![Python](https://img.shields.io/badge/Python-3.8-blueviolet)
![Framework](https://img.shields.io/badge/Framework-Flask-red)
![API](https://img.shields.io/badge/API-TMDB-fcba03)

A content-based recommender system that recommends movies similar to the movie selected by the user.


## Overview

The movies are recommended based on the content of the movie you entered or selected. The main parameters that are considered for the recommendations are the genre, director, and top 3 casts. The details of the movies, such as title, genre, runtime, rating, poster, casts, etc., are fetched from [TMDB](https://www.themoviedb.org/documentation/api).


![working image](https://user-images.githubusercontent.com/66157952/208297696-a6be66fd-acf0-4edd-b368-6b0f339637bd.png)

## How to get the API key?

Create an account in https://www.themoviedb.org/. Once you successfully created an account, click on the `API` link from the left hand sidebar in your account settings and fill all the details to apply for an API key. If you are asked for the website URL, just give "NA" if you don't have one. You will see the API key in your `API` sidebar once your request has been approved.

## How to run the project?

1. Clone or download this repository to your local machine.
2. Get your API key from https://www.themoviedb.org/. (Refer the above section on how to get the API key)
3. Replace YOUR_API_KEY in **both** the places (line no. 15 and 29) of `static/recommend.js` file and hit save.
4. Open your terminal/command prompt from your project directory and run the file `main.py` by executing the command `python main.py`.
5. Go to your browser and type `http://127.0.0.1:5000/` in the address bar.
6. Hurray! That's it.


