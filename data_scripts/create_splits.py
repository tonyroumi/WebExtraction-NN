import os
import sys
from torch.utils.data import DataLoader, SubsetRandomSampler
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from tools.utils import *


RESULT_PATH = 'data_news/page_sets/splits/'
PAGE_SETS_PATH = 'data_news/page_sets/'
SHOP_LIST_PATH = 'data_news/shop_list.txt'
PRIORS_DIRECTORY = '../data_shops/split_priors/'

def KFold(n, n_folds):
    indices = list(range(n))
    fold_size = n // n_folds
    remainder = n % n_folds

    folds = []
    start = 0
    for i in range(n_folds):
        fold_len = fold_size + 1 if i < remainder else fold_size
        fold_indices = indices[start:start + fold_len]
        test_indices = indices[:start] + indices[start + fold_len:]
        folds.append((fold_indices, test_indices))
        start += fold_len

    return folds

def getPagesForShops(shops):
    pages = []
    for shop in shops:
        page_set_path = os.path.join(PAGE_SETS_PATH,shop)
        with open(page_set_path, 'r') as f:
            shop_pages = [l.strip() for l in f.readlines()]
            pages.extend(shop_pages)

    return pages


def createListFile(filename, pages):
    # for each page
    lines = []
    for page in pages:
        line=page
        lines.append(line)

    with open(os.path.join(RESULT_PATH, filename),'w') as f:
        for line in lines:
            f.write(line+'\n')




#----- MAIN PART
if __name__ == "__main__":

    if not os.path.exists(PRIORS_DIRECTORY):
        os.makedirs(PRIORS_DIRECTORY)

    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    if os.path.exists(SHOP_LIST_PATH):
      os.remove(SHOP_LIST_PATH)

    with open(SHOP_LIST_PATH, 'a') as shop_list_file:
        for filename in os.listdir(PAGE_SETS_PATH):
            if filename.endswith(".txt"):
                with open(os.path.join(PAGE_SETS_PATH, filename), 'r') as txt_file:
                    for line in txt_file:
                        shop_list_file.write(line)
  
    # read shop list
    with open(SHOP_LIST_PATH, 'r') as f:
        shops = [l.strip() for l in f.readlines()]

    kf = KFold(len(shops), n_folds=10)
    
    split_num=1
    for val_index, train_index in kf:
        training = [shops[i] for i in train_index]
        validation = [shops[i] for i in val_index]

       
  
        createListFile('split_'+str(split_num)+'_train.txt',training)
        createListFile('split_'+str(split_num)+'_val.txt',validation)
        # create_position_maps(training+validation, split_num)
        

        split_num+=1

