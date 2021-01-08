import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE 
from scipy.spatial.distance import cdist

st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)
st.title('Find the Right Skin Care for you')

st.write("Hi there! :wave: If you have a skincare product you currently like I can help you find a similar one based on the ingredients. :point_up:")

st.write('Please select a product below so I can recommend similar ones :relieved:')
st.write('My dataset contains 1400+ products :star2: but unfortunately it is possible that I do not have the product you are looking for :disappointed:')
# Load the data
df = pd.read_csv("./data/cosmetics.csv")

# Choose a product category
category = st.selectbox(label='Select a product category', options= df['Label'].unique() )
category_subset = df[df['Label'] == category]
# Choose a brand
brand = st.selectbox(label='Select a brand', options= sorted(category_subset['Brand'].unique()))
category_brand_subset = category_subset[category_subset['Brand'] == brand]
# Choose product
product = st.selectbox(label='Select the product', options= sorted(category_brand_subset['Name'].unique() ))

#skin_type = st.selectbox(label='Select your skin type', options= ['Combination',
#       'Dry', 'Normal', 'Oily', 'Sensitive'] )

## Helper functions
# Define the oh_encoder function
def oh_encoder(tokens):
    x = np.zeros(N)
    for ingredient in tokens:
        # Get the index for each ingredient
        idx = ingredient_idx[ingredient]
        # Put 1 at the corresponding indices
        x[idx] = 1
    return x

def closest_point(point, points):
    """ Find closest point from a list of points. """
    return points[cdist([point], points).argmin()]


if category is not None:
    category_subset = df[df['Label'] == category]

if product is not None:
    #skincare_type = category_subset[category_subset[str(skin_type)] == 1]

    # Reset index
    category_subset = category_subset.reset_index(drop=True)

    # Display data frame
    #st.dataframe(category_subset)

    # Initialize dictionary, list, and initial index
    ingredient_idx = {}
    corpus = []
    idx = 0

    # For loop for tokenization
    for i in range(len(category_subset)):    
        ingredients = category_subset['Ingredients'][i]
        ingredients_lower = ingredients.lower()
        tokens = ingredients_lower.split(', ')
        corpus.append(tokens)
        for ingredient in tokens:
            if ingredient not in ingredient_idx:
                ingredient_idx[ingredient] = idx
                idx += 1

                
    # Get the number of items and tokens 
    M = len(category_subset)
    N = len(ingredient_idx)

    # Initialize a matrix of zeros
    A = np.zeros((M,N))

    # Make a document-term matrix
    i = 0
    for tokens in corpus:
        A[i, :] = oh_encoder(tokens)
        i +=1

model_run = st.button('Find similar products!')


if model_run:

    st.write('Based on the ingredients of the product you selected')
    st.write('here are the top 10 products that are the most similar :sparkles:')
    
    # Run the model
    model = TSNE(n_components = 2, learning_rate = 150, random_state = 42)
    tsne_features = model.fit_transform(A)

    # Make X, Y columns 
    category_subset['X'] = tsne_features[:, 0]
    category_subset['Y'] = tsne_features[:, 1]

    target = category_subset[category_subset['Name'] == product]

    target_x = target['X'].values[0]
    target_y = target['Y'].values[0]

    df1 = pd.DataFrame()
    df1['point'] = [(x, y) for x,y in zip(category_subset['X'], category_subset['Y'])]

    category_subset['distance'] = [cdist(np.array([[target_x,target_y]]), np.array([product]), metric='euclidean') for product in df1['point']]

    # arrange by descending order
    top_matches = category_subset.sort_values(by=['distance'])

    # Compute ingredients in common
    target_ingredients = target.Ingredients.values
    c1_list = target_ingredients[0].split(",")
    c1_list = [x.strip(' ') for x in c1_list]
    c1_set = set(c1_list)

    top_matches['Ingredients in common'] = [c1_set.intersection( set([x.strip(' ')for x in product.split(",")]) ) for product in top_matches['Ingredients']]

    # Select relevant columns
    top_matches = top_matches[['Label', 'Brand', 'Name', 'Price', 'Ingredients','Ingredients in common']]
    top_matches = top_matches.reset_index(drop=True)
    top_matches = top_matches.drop(top_matches.index[0])

    st.dataframe(top_matches.head(10))


