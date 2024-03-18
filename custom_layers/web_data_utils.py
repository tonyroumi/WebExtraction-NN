import re
import numpy as np
import mmh3
import torch.nn.functional as F
import torch


def preprocess_string(text):
    res = re.sub("[!.;?^*()_{}|]","", text) # remove special characters (we keep characters such as "$" and ",-" )
    res = re.sub("\d+", " ^number^ ", res)   # replace numbers wit special word and space
    return res

def my_tokenizer(s):
    return s.split()

def hash_vectorizer(texts, n_features):
    # Define a dictionary to store the hashed vectors
    hashed_vectors = []

        # Preprocess the text
    preprocessed_text = preprocess_string(texts)
        
        # Tokenize the text
    tokens = my_tokenizer(preprocessed_text)
        
        # Initialize a vector with zeros
    vector = np.zeros(n_features)

        # Hash each token and update the corresponding index in the vector
    for token in tokens:
            # Hash the token
            
            # Convert the hashed token to an integer
        hashed_index = mmh3.hash(token) % n_features
            
            # Increment the corresponding index in the vector
        vector[hashed_index] += 1

        # Append the hashed vector to the list of hashed vectors
    hashed_vectors.append(vector)

    return np.array(hashed_vectors)


def get_text_nodes(leaf_nodes, n_features):
    text_nodes = []

    for node in leaf_nodes:
        #-- process text nodes
        # if it is text node with value
        if 'type' in node and 'value' in node:
            position = node['position']
            
            size = [(position[2]-position[0])*(position[3]-position[1])]
              
            # get text - remove whitespaces, lowercase
            text = node['value']
            text = ' '.join(text.lower().split())
            encoded_text = hash_vectorizer(text, n_features)

            if len(encoded_text.nonzero()[0]) > 0:
                text_nodes.append((position,encoded_text,size))


    # ORDER TEXT NODES BY SIZE
    text_nodes.sort(key=lambda x: x[2], reverse=True)  

    return text_nodes

def get_text_maps(text_nodes, n_features, spatial_shape, text_map_scale):
    # scale down spatial dimensions
    features = np.zeros((round((spatial_shape[0]*text_map_scale)),round((spatial_shape[1]*text_map_scale)), n_features), dtype=np.float32)
    
    # for each node in text nodes
    for node in text_nodes:
        bb = node[0]
        bb_scaled = [int(round(x*text_map_scale)) for x in bb]
        encoded_text = node[1]
        encoded_text = (torch.from_numpy(encoded_text)).float()
        encoded_text = F.normalize(encoded_text, p=2, dim=1)
        encoded_text = encoded_text*255   # we multiply by 255 in order to move to image scale
        vector = encoded_text[0]
        features[bb_scaled[1]:bb_scaled[3],bb_scaled[0]:bb_scaled[2],:] = vector
    return features

