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
import lxml


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

    
    # Initialize Selenium WebDriver
    chrome_options = Options()
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
                 else:
                    textNode = createNode(i)
                    node['childNodes'].insert(0, textNode)

                 
        #    processed_stack.append(node)
        return processed_stack.pop()
        
           
                   
           
    
    # Function to render next URL
    def retrieve():
        nonlocal urlIndex
        if (urlIndex < len(urls)):
             
             #Possibly add some logic to skip over already downloaded files

            url = urls[urlIndex]
            urlIndex += 1
              
            driver.get(url)
            time.sleep(4) 
              
              # Get paths
            image_path = getImagePath(urlIndex)
            pageID = getPageID(urlIndex)
            dom_tree_path = getDOMPath(urlIndex)
            html_path = getHTMLPath(urlIndex)
            list_path = getListPath()
            try:
               html_content = driver.page_source
               dom_tree = getDOMTree()
               driver.save_screenshot("screenshot.png")
               image = Image.open("screenshot.png")
               image = image.convert("RGB")
               image.save(image_path, format="JPEG", quality=100) 
               callbackPerUrl("success", url,pageID, dom_tree_path, html_path, list_path, dom_tree, html_content)         
            
            
            except:
              callbackPerUrl("failure", url, pageID, dom_tree_path, html_path, list_path, None, None)
            retrieve()
        else:
            callbackFinal(driver)
    
    listPath = getListPath()
    if not os.path.exists(listPath):
       os.makedirs(listPath)
   
        

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
       print("Unable to render" + url + "OOPS")
       

# Callback function after all URLs have been processed
def callbackFinal(driver):
    print("All URLs processed.")
    driver.quit()

# Main function
if __name__ == "__main__":
    if (len(sys.argv) == 2):
        prefix = sys.argv[1]
        input_path = "data_news/"+prefix+".txt"
        output_path = "data_news"
    else: 
        print("Usage: python download_news.py TRAININGFILENAME")
        
    
    # Read URLs from file
    with open(input_path) as f:
        urls = f.read().splitlines()
        f.close()
        

    # Run rendering
    RenderUrlsToFile(urls, output_path, prefix, callbackPerUrl, callbackFinal)
