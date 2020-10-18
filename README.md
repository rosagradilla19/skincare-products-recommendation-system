# Reccommendation System for Skin Care Products based on Ingredients

For this project I wanted to develop a recommendation system for skin care products based on ingredients.

Although some consumers might be attracted to a product based on the brand or marketing techniques I am a firm believer that ingredients are the most important thing to consider when purchasing a skin care product. I am personally very interested in skin care and whenever I wanted to try a new product I would find myself looking at the ingredient list, googling every ingredient I did not recognize. This is of course very time consuming. I decided to create a recommendation system based on ingredients.

To find a low-dimensional representation of the products I used a t-distributed stochastic neighbor embedding (tSTNE). This embedding takes the high dimensional data into a 2D representation which is used to find similar products.

The data contains more than 1,400 products in 5 different categories:

- Moisturizer
- Cleanser
- Face mask
- Treatment
- Eye cream
- Sun protection
