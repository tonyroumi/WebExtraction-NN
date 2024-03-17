# Scripts used for downloading webpages

## Software needed

- BeautifulSoup4
- Selenium
- python

## Create a folder named sources and a text file in sources with the name of a specfiic source
Example: thetech.txt

## Download webpages pages from publication
Run following script:

```Shell
python download_shop.py [PREFIX]
#example: python download_shop.py thetech) 
```

## Semi-automatic labeling of DOM elements 
Run following script:

python labelling.py [prefix]

Script creates new directory "labeled_dom_trees" which contains copy of DOM trees with labeled elements.

## Review labeled results

We review labeled results by checking image patches of labeled elements. The process is divided into 3 steps - prepare labeled patches, review them, remove them.

### Step 1: Prepare patches
```Shell
python review.py prepare [prefix]
```

### Step 2: Review patches
```Shell
python review.py review [prefix]
```

You can select wrongly labeled patches, in order to remove page from dataset. If everything goes right, the script creates new file in "page_sets" directory,
which contains all pages that passed the review process.

### Step 3: Remove patches
```Shell
python review.py remove [prefix]
```

## Create boxes and text maps that enter neural net
```Shell
python create_net_inputs.py [prefix]
```
