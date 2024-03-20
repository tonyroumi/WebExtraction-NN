import os
import cv2
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

IMAGES_PER_LINE = 10
MAX_LINES = 10

FIG_HEIGHT = 8 
FIG_WIDTH =  8


MAX_PATCH_SIZE = 100.0

LABELS = ['title','date','content']

DOM_PATH = '../data_news/labeled_dom_trees/'
PATCHES_PATH = '../data_news/review_patches/'
DOWNLOADED_PAGES_PATH = '../data_news/downloaded_pages/'
PAGE_SETS_PATH = '../data_news/page_sets/'

PAGES_TO_DELETE = set()

# find labeled elements
def getLabeledElements(dom_path):
    results = {}

    with open(dom_path,'r') as f:
            root = json.load(f)

    processing_stack = []
    processing_stack.append(root)
    while len(processing_stack)!=0:
        node = processing_stack.pop()

        # get label
        if 'label' in node:
            label = node['label']
            results[label] = node
 
        # follow children
        if 'childNodes' in node:
            childNodes = node['childNodes']
            for childNode in childNodes:
                processing_stack.append(childNode)

    return results

def getPatch(im, element):
    #L,T,R,B in pixels 

    position = element['position']
    im_height, im_width, _ = im.shape
    patch_top = max(0,position[1])
    patch_left = max(0, position[0]) 
    patch_bottom = min(im_height, position[3])  
    patch_right = min(im_width, position[2]) 

    cropped_patch = im[patch_top:patch_bottom, patch_left:patch_right, :]

    
    
    return cropped_patch

def getLabeledPageList(prefix):
    # load all pages
    pages_path = os.path.join(DOWNLOADED_PAGES_PATH, prefix+'.txt') 
    with open(pages_path,'r') as f:
        pages = [line.split('\t')[0] for line in f.readlines()]

   # get labeled pages
    labelled_pages = []
    for page in pages:
        dom_path = os.path.join(DOM_PATH, page+'.json')

        # if we have label add it
        if os.path.isfile(dom_path):
            labelled_pages.append(page)
  
    return labelled_pages


def preparePatches(prefix):
    # create patches directory if it does not exist
    if not os.path.exists(PATCHES_PATH):
        os.makedirs(PATCHES_PATH)


    # load pages
    pages = getLabeledPageList(prefix)

    # for each page from prexix
    for page in pages:
        print('Creating patches for:' + str(page))

        # prepare paths
        dom_path = os.path.join(DOM_PATH, page+'.json')
        page_image_path = os.path.join('../data_news/images/', page+'.jpeg')
        
        # if we have labeled version
        if os.path.isfile(dom_path):
            # get labeled elements
            labeled = getLabeledElements(dom_path)

            # load image
            im = cv2.imread(page_image_path)
            for i in range(len(LABELS)):
                label = LABELS[i]
                patch = getPatch(im,labeled[label])
                
                # # get edge size
                # edge_size = np.max(patch.shape[:2])
               
                # # # if edge is too big -> update patch
                # if edge_size>MAX_PATCH_SIZE:
                #     ratio = MAX_PATCH_SIZE/edge_size
                #     patch = cv2.resize(patch,(0,0), fx=ratio, fy=ratio, interpolation = cv2.INTER_LINEAR)
                    
                # save
                try:
                    path = os.path.join(PATCHES_PATH, page+'_'+label+'.jpeg')
                    cv2.imwrite(path,patch)
                except:
                    print("Patch failed for: " + str(page) + " Out of image bounds")

def onPick(event):
    # # new selected patch
    selected_patch = event.artist

    if selected_patch:
        page = selected_patch.page
        
        if page not in PAGES_TO_DELETE:
            PAGES_TO_DELETE.add(page)
            selected_patch.set_linewidth(4)
            selected_patch.set_edgecolor('red')
            print('*'+ str(page)+ ' will be removed')

        else:
            PAGES_TO_DELETE.remove(page)
            selected_patch.set_linewidth(1)
            selected_patch.set_edgecolor('black')
            print('*'+ str(page)+ ' will not be removed')
    
        selected_patch.figure.canvas.draw()


def keyPress(event):
    if event.key == 'enter':
        plt.close()

def reviewPatches(prefix):
    # load pages
    pages = getLabeledPageList(prefix)
    batch_size = IMAGES_PER_LINE*MAX_LINES

    # for each label type
    for label in LABELS:
        print('Please, review label: ' + label)
        ind = 0

        # for every page
        while ind+1<len(pages):
            fig = plt.figure(figsize=(FIG_WIDTH,FIG_HEIGHT))
            ax = fig.add_subplot(111)
            ax.set_facecolor('grey')
           

            # ax.set_autoscaley_on(False)
            # ax.set_autoscalex_on(False)
            ax.set_ylim([0,10])
            ax.set_xlim([0,10])

            # put in plot
            for row in range(MAX_LINES):
                for col in range(IMAGES_PER_LINE):

                    if ind<len(pages):
                        show_page = pages[ind]
                        labeled_dom_path = os.path.join(DOM_PATH, show_page+'.json')

                        # if we have labeled version
                        if os.path.isfile(labeled_dom_path):
                            path_to_patch = os.path.join(PATCHES_PATH,show_page+'_'+label+'.jpeg')
                            patch = cv2.imread(path_to_patch)
                            
                            if patch is not None:                            
                                ax.imshow(patch, extent=[col,col+1,row,row+1], aspect="auto")
                                rect = plt.Rectangle((col,row),1, 1, facecolor=(0,0,0,0),picker=5)
                                rect.page = show_page
                                ax.add_patch(rect)
                            else:
                                PAGES_TO_DELETE.add(show_page)
                        # if we have not we will delete it
                        else:
                            PAGES_TO_DELETE.add(show_page)

                    ind = ind+1

            fig.canvas.mpl_connect('pick_event', onPick)
            fig.canvas.mpl_connect('key_press_event', keyPress)
            plt.show()

    # if result directory does not exist, create it
    if not os.path.exists(PAGE_SETS_PATH):
        os.makedirs(PAGE_SETS_PATH)


    # save result
    with open(os.path.join(PAGE_SETS_PATH,prefix+'.txt'),'w+') as f:
        for page in pages:
            if page not in PAGES_TO_DELETE:
                f.write(page+'\n')


def removePatches(prefix):
    # load pages
    pages = getLabeledPageList(prefix)

    # for each page from prexix
    for page in pages:
        print('Removing: '+ str(page))

        # for each label
        for i in range(len(LABELS)):
            label = LABELS[i]
            patch_path = os.path.join(PATCHES_PATH, page+'_'+label+'.jpeg')

            # if it exists remove
            if os.path.isfile(patch_path):
                os.remove(patch_path)

#----- MAIN PART
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', type=str, choices=['prepare', 'review', 'remove'], help='phase of review process')
    parser.add_argument('prefix', type=str, help='prefix of eshop')
    args = parser.parse_args()
    
    # if phase is to prepare 
    if args.phase == 'prepare':
        preparePatches(args.prefix)

    if args.phase == 'review':
        reviewPatches(args.prefix)

    if args.phase == 'remove':
        removePatches(args.prefix)
