import os
import cv2
import sys
import json
import copy
import pickle
import random
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from custom_layers.dom_tree import DOMTree
import matplotlib.pyplot as plt


FIGURE_WIDTH = 14
FIGURE_HEIGHT = 10
MAX_LABELED = 500

DOWNLOADED_PAGES_LIST_PATH = '../data_news/downloaded_pages/'
LABELED_DOM_PATH =  '../data_news/labeled_dom_trees/'
IMAGES_PATH = '../data_news/images/'
DOM_PATH = '../data_news/dom_trees/'
PATHS_PATH = '../data_news/element_paths/'

#----- CLASS FOR SELECTING A NODE
class ElementSelector:

    selected_patch = None

    def __init__(self, image_path, dom_tree):
        self.image_path = image_path
        self.dom_tree = dom_tree

    def onPick(self, event):
        # change back old selected page
        if self.selected_patch:
            self.selected_patch.set_linewidth(1)

        # new selected patch
        self.selected_patch = event.artist

        # draw
        self.selected_patch.set_linewidth(4)
        self.selected_patch.figure.canvas.draw()


    def keyPress(self, event):
        if event.key == 'enter':
            plt.close()

    def selectElement(self):
        ## CROP IMAGE
        crop_top = 900
        self.fig = plt.figure(figsize=(FIGURE_WIDTH,FIGURE_HEIGHT))
        im = cv2.imread(self.image_path)
        
        im = im[:crop_top,:,:]        

        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

        patches = []

         #L,T,R,B in pixels 

        # for each leaf node
        for leafNode in self.dom_tree.getPositionedLeafNodes():
            
            position = leafNode['position']

            # Scale and adjust the coordinates since screenshot taken is fullsize
            position = [x * 2 for x in position]

            # text nodes have different color (just for sanity checks)
            patch = plt.Rectangle((position[0], position[1]) ,position[2]-position[0],position[3]-position[1], fill=False, edgecolor='g' if 'type' in leafNode else 'b', linewidth=1, picker=3)


            patch.node = leafNode

            # compute size
            size = (position[2]-position[0])*(position[3]-position[1])
            # add to patch list
            patches.append((patch,size))

        #Include later
        # patches = [(patch, size) for (patch, size) in patches if patch[1] >= self.dom_tree['html']['random_scroll']]  # Filter patches starting below the scroll position
        patches.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place
        for (patch,size) in patches:
            plt.gca().add_patch(patch)

        self.fig.canvas.mpl_connect('pick_event', self.onPick)
        self.fig.canvas.mpl_connect('key_press_event', self.keyPress)
        plt.show()


        if self.selected_patch:
            return self.selected_patch.node
        else:
            return None

def findAndLabel(page, authorPaths, datePaths, contentPaths):
    # get dom
    dom = getPageDOM(page)

    # find elements
    authorElement = dom.getElementByOneOfPaths(authorPaths)
    dateElement = dom.getElementByOneOfPaths(datePaths)
    contentElement = dom.getElementByOneOfPaths(contentPaths)

    # if we have all elements
    if authorElement and dateElement and contentElement:
        authorElement['label'] = 'author'
        dateElement['label'] = 'date'
        contentElement['label'] = 'content'

        dom.saveTree(os.path.join(LABELED_DOM_PATH, page+'.json'))
        return True

    # if we do not have all
    else:
        return False

def loadPaths(prefix):
    print ('Loading paths')
    authorPaths = []
    datePaths = []
    contentPaths = []
    path_to_saved_path = os.path.join(PATHS_PATH, prefix+'.pkl')
    if os.path.exists(path_to_saved_path):
        paths = pickle.load(open(path_to_saved_path,'rb'))
        authorPaths = paths['author']
        datePaths = paths['date']
        contentPaths = paths['content']

    return authorPaths, datePaths, contentPaths

def savePaths(prefix, authorPaths, datePaths, contentPaths):
    paths = {}
    paths['author'] = authorPaths
    paths['date'] = datePaths
    paths['content'] = contentPaths

    #There are all being written to the same file , not sure if that's what we want to do
    #Or Paths path will be the same, but different prefixes for different sites
     
    pathToSavedPath = os.path.join(PATHS_PATH, prefix+'.pkl')
    pickle.dump(paths, open(pathToSavedPath,'wb+'))


def getUnlabeledPages(pages):
    print ('Getting unlabed pages')

    unlabeled = []
    labeledCount=0
    for page in pages:
        path = os.path.join(LABELED_DOM_PATH,page+'.json')
        if os.path.exists(path):
            labeledCount+=1
        else:
            unlabeled.append(page)
            
    print('Unlabeled count:'+ str(len(unlabeled)))
    print('Labeled count:'+ str(labeledCount))
    return unlabeled

def getPageDOM(page):
    dom_path = os.path.join(DOM_PATH,page+'.json')
    return DOMTree(dom_path)

def selectNewPaths(image_path, dom):
    selector = ElementSelector(image_path,dom)
    element = selector.selectElement()
    if element:
        return dom.getPaths(element)
    else:
        return []


def getNewPaths(pages, authorPaths, datePaths, contentPaths):
    updatedPath = False

    # until we have no updated path
    while not updatedPath:
        random_page = random.choice(pages)
        dom = getPageDOM(random_page)
        page_image_path = os.path.join(IMAGES_PATH,random_page+'.jpeg')
        displayQuestion=True

        newAuthorPaths = []
        newDatePaths = []
        newContentPaths = [] 

        # try to get author
        authorElement = dom.getElementByOneOfPaths(authorPaths)
        if authorElement is None and displayQuestion:
            print('Help me to find the author:')
            newAuthorPaths = selectNewPaths(page_image_path, dom)
            if len(newAuthorPaths)>0:
                updatedPath=True
            else:
                displayQuestion=False

        # try to get date
        dateElement = dom.getElementByOneOfPaths(datePaths)
        if dateElement is None and displayQuestion:
            print('Help me to find the date:')
            newDatePaths = selectNewPaths(page_image_path, dom)
            if len(newDatePaths)>0:
                updatedPath=True
            else:
                displayQuestion=False

        # try to get content
        contentElement = dom.getElementByOneOfPaths(contentPaths)
        if contentElement is None and displayQuestion:
            print('Help me to find the content:')
            newContentPaths = selectNewPaths(page_image_path, dom)
            if len(newContentPaths)>0:
                updatedPath=True
            else:
                displayQuestion=False

    return newAuthorPaths, newDatePaths, newContentPaths


if __name__ == "__main__":

    # read params
    if len(sys.argv) != 2:
        print('BAD PARAMS. USAGE [prefix]')
        sys.exit(1)
    
    prefix = sys.argv[1]

    # prepare output path
    if not os.path.exists(LABELED_DOM_PATH):
        os.makedirs(LABELED_DOM_PATH)
    if not os.path.exists(PATHS_PATH):
        os.makedirs(PATHS_PATH)

    # load pages
    pagesPath = os.path.join(DOWNLOADED_PAGES_LIST_PATH, prefix+ '.txt')
    with open(pagesPath,'r') as f:
        pages = [line.split('\t')[0] for line in f.readlines()]

    # try to load paths to elements
    authorPaths, datePaths, contentPaths = loadPaths(prefix)
    
    # split pages to already labeled or unlabeled
    unlabeledPages = getUnlabeledPages(pages)

    totalLabeledCount = len(pages)-len(unlabeledPages)

    # until there are some unlabeled_pages
    while len(unlabeledPages)>0:

        # get new paths
        print('Get new paths')
        newAuthorPaths, newDatePaths, newContentPaths  = getNewPaths(unlabeledPages, authorPaths, datePaths, contentPaths)

        # update existing paths
        print('Updating paths')
        authorPaths.extend(newAuthorPaths)
        datePaths.extend(newDatePaths)
        contentPaths.extend(newContentPaths)

        # save new updated paths
        print('Saving new paths')
        savePaths(prefix,  authorPaths, datePaths, contentPaths)

        # try to annotate page
        print('Annotating other pages')

        #succeded_count = 0
        newUnlabeledPages = []

        for page in unlabeledPages:
            success = findAndLabel(page, authorPaths, datePaths, contentPaths)
            if success:
              print(str(totalLabeledCount) + " " + str(page))
              totalLabeledCount+=1
            else:
                newUnlabeledPages.append(page)

            if totalLabeledCount>=MAX_LABELED:
                break

        # print result
        print("Successfully labeled: "+ str(totalLabeledCount) + 'pages.')
        print("Unlabeled pages: " + str(len(newUnlabeledPages))+ 'pages.')
   
        unlabeledPages = newUnlabeledPages
        
        # check maximum threshold
        if totalLabeledCount>=MAX_LABELED:
            print("Maximum number of labeled examples achieved")
            break
