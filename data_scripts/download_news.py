import os
import json
import time
from selenium import webdriver
from bs4 import BeautifulSoup as bs
import sys
from PIL import Image
from io import BytesIO
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities


DOWNLOADED_PAGES_PATH = '../data_news/downloaded_pages/'
DOM_PATH = '../data_news/dom_trees/'

def xpath_soup(element):
   components = []
   target = element if element.name else element.parent
   for node in (target, *target.parents)[-2::-1]:  # type: bs4.element.Tag
      tag = '%s:%s' % (node.prefix, node.name) if node.prefix else node.name
      siblings = node.parent.find_all(tag, recursive=False)
      components.append(tag if len(siblings) == 1 else '%s[%d]' % (tag, next(
            index
            for index, sibling in enumerate(siblings, 1)
            if sibling is node
            )))
   return '/%s' % '/'.join(components)

# Function to render URLs to file
def RenderUrlsToFile(urls, output_path, prefix, callbackPerUrl, callbackFinal):
    urlIndex = 0

    
    # Initialize Selenium WebDriver, disable popups and images
    chrome_options = Options()
    chrome_options.add_argument("load-extension=../Extensions/adblock.crx")
    # chrome_options.add_argument('--blink-settings=imagesEnabled=false')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('window-size=1280x800')
    driver = webdriver.Chrome(options=chrome_options)
    

    
    
    # Function to get image path
    def getImagePath(urlIndex):
        return os.path.join(output_path, "images", f"{getPageID(urlIndex)}.jpeg")

    # Function to get page ID
    def getPageID(urlIndex):
        return f"{prefix}-{str(urlIndex).zfill(6)}"

    # Function to get DOM path
    def getDOMPath(urlIndex):
        return os.path.join(output_path, "dom_trees", f"{getPageID(urlIndex)}.json")

    # Function to get HTML path
    def getHTMLPath(urlIndex):
        return os.path.join(output_path, "htmls", f"{getPageID(urlIndex)}.html")
    
    def getListPath():
        return os.path.join(output_path, "downloaded_pages",prefix+".txt")

    # Function to retrieve DOM tree
    def getDOMTree():
        def getElements():
           soup = bs(driver.page_source, 'lxml')
           return soup.find_all()
        
        def createNode(element): #individual element tag and contents
           node = {}
           if(element.name == None):
              node['type'] = 3
              node['name'] = 'text'
           else:
              node['name'] = element.name
           
           
           if(element.string != None):
              node['value'] = element.string
           else:
              node['value'] = element.text
           xpath = xpath_soup(element)
           driver.implicitly_wait(2)
           try:
            elem = driver.find_element('xpath', xpath)
        # Find the corresponding WebDriver element using XPath

            computed_style = driver.execute_script('''
            var elem = arguments[0];
            var computedStyles = window.getComputedStyle(elem);
            var selected_style_props = ['display','visibility','opacity','z-index','background-image','content','image'];
            var styles = {};
            if(computedStyles) {
                for(var i=0; i < selected_style_props.length; i++) {
                    styles[selected_style_props[i]] = computedStyles.getPropertyValue(selected_style_props[i]);
                }
            }
            return styles;''', elem)
            
            if(computed_style):
                node['computed_style'] = computed_style
            location = elem.location
            
            size = elem.size
            #Position L,T,R,B in pixels 
            node['position'] = [location['x'], location['y'], location['x']+size['width'], location['y']+size['height']]
            attrs = driver.execute_script('''
                var elem = arguments[0];
                var attributes = elem.attributes;
                var attributes_dict = {};
                if(attributes)
                for (var i = 0; i < attributes.length; i++) {
                    var attr = attributes[i];
                    attributes_dict[attr.nodeName] = attr.nodeValue;
                    }
                return attributes_dict;''', elem)
            if(attrs):
                node['attrs'] = attrs
           except:
              node['trash'] = True
              return node
           return node
        
        element_stack = getElements()
        processed_stack = []

        while(len(element_stack) != 0):
           element = element_stack.pop()
           node = createNode(element)
           if(len(element.contents) != 0):
              node['childNodes']= []
              for i in element.contents:
                 if(i.name is not None): 
                    childNode = processed_stack.pop() 
                    node['childNodes'].insert(0, childNode)
                 elif(i.text.strip() and i.text != "\n"):
                    textNode = createNode(i)
                    node['childNodes'].insert(0, textNode)
           processed_stack.append(node)
        return processed_stack.pop()
    
    
    # Function to render next URL
    def retrieve():
        nonlocal urlIndex
        if (urlIndex < len(urls)):
            url = urls[urlIndex]
            urlIndex += 1
              
            
              
              # Get paths
            image_path = getImagePath(urlIndex)
            pageID = getPageID(urlIndex)
            dom_tree_path = getDOMPath(urlIndex)
            html_path = getHTMLPath(urlIndex)
            list_path = getListPath()
            try:
               driver.get(url)
               sc = driver.get_screenshot_as_png()
               html_content = driver.page_source
               dom_tree = getDOMTree()
               sc = driver.get_screenshot_as_png()
               Image.open(BytesIO(sc)).resize((1280,800)).convert("RGB").save(image_path, format="JPEG", quality=100)

               callbackPerUrl("success", url,pageID, dom_tree_path, html_path, list_path, dom_tree, html_content)         
            
            
            except:
              callbackPerUrl("failure", url, pageID, dom_tree_path, html_path, list_path, None, None)
            retrieve()
        else:
            callbackFinal(driver)

    #Check to see if pages have already been downloaded
    pages_path = os.path.join(DOWNLOADED_PAGES_PATH, prefix+'.txt') 
    if os.path.exists(pages_path):
        with open(pages_path,'r') as f:
            pages = [line.split('\t')[0] for line in f.readlines()]
        for page in pages:
            dom_path = os.path.join(DOM_PATH, page+'.json')
            if os.path.isfile(dom_path):
                urlIndex += 1

    retrieve()

def callbackPerUrl(status, url, pageID, dom_tree_path, html_path, listPath, dom_tree, html):
    if status == "success":
      print(url)
      if not os.path.exists(listPath):
        with open(listPath, 'w') as f:
          f.write(pageID + "\t" + url)
      else:
        with open(listPath, 'a') as f:
          f.write("\n" + pageID + "\t" + url)
      dom_content = json.dumps(dom_tree, indent=4, default=str)
      with open(dom_tree_path, 'w') as f:
         f.write(dom_content)
      with open(html_path, 'w') as f:
         f.write(html)
    else:
       print("Unable to render: " + url + " OOPS")
       

# Callback function after all URLs have been processed
def callbackFinal(driver):
    print("All URLs processed.")
    driver.quit()

# Main function
if __name__ == "__main__":
    if (len(sys.argv) == 2):
        prefix = sys.argv[1]
        input_path = "../data_news/sources/"+prefix+".txt"
        output_path = "../data_news"
    else: 
        print("Usage: python download_news.py TRAININGFILENAME")
        
    
    # Read URLs from file
    with open(input_path) as f:
        urls = f.read().splitlines()
        f.close()
        

    # Run rendering
    RenderUrlsToFile(urls, output_path, prefix, callbackPerUrl, callbackFinal)


