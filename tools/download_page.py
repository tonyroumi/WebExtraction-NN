import os
import json
import time
from selenium import webdriver
from bs4 import BeautifulSoup as bs
import sys
from PIL import Image
from io import BytesIO
from selenium.webdriver.chrome.options import Options
import requests

DOWNLOADED_PAGES_PATH = 'data_news/downloaded_pages/'
DOM_PATH = 'data_news/dom_trees/'

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
def RenderUrl(url, output_path):
    urlIndex = 0

    
    # Initialize Selenium WebDriver with adblocker and don't allow images
    chrome_options = Options()
    chrome_options.add_argument("load-extension=../Extensions/adblock.crx")
   #  chrome_options.add_argument('--blink-settings=imagesEnabled=false')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('window-size=1280x800')
    driver = webdriver.Chrome(options=chrome_options)

    def saveDomTree(dom_tree_path, dom_tree):
        dom_content = json.dumps(dom_tree, indent=4, default=str)
        with open(dom_tree_path, 'w') as f:
           f.write(dom_content)
    
    
    # Function to get image path
    def getImagePath(output_path):
        return os.path.join(output_path, "screenshot.jpeg")


    # Function to get DOM path
    def getDOMPath(output_path):
        return os.path.join(output_path, "dom.json")

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
  
    driver.get(url)
    time.sleep(4) 
              
             
    image_path = getImagePath(output_path)        
    dom_tree_path = getDOMPath(output_path)
    print(image_path)

    try:
      dom_tree = getDOMTree()
      sc = driver.get_screenshot_as_png()   
      Image.open(BytesIO(sc)).resize((1280,800)).convert("RGB").save(image_path, format="JPEG", quality=100)
      saveDomTree(dom_tree_path, dom_tree)
      print("Success")
      driver.quit()
    except:
      print("Unsuccessful")
      driver.quit()


if __name__ == "__main__":
   if (len(sys.argv) == 3):
      url = sys.argv[1]
      output_path = sys.argv[2]
   else: 
      print("Usage: python download_news.py URL OUTPUT_PATH")
         
      # Run rendering
   RenderUrl(url, output_path)


