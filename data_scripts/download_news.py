import os
import json
import time
from selenium import webdriver
from bs4 import BeautifulSoup as bs
import sys
from PIL import Image
from io import BytesIO

# Function to render URLs to file
def RenderUrlsToFile(urls, output_path, prefix, callbackPerUrl, callbackFinal):
    urlIndex = 0
    
    # Initialize Selenium WebDriver
    driver = webdriver.Chrome()
    driver.set_window_size(1280, 800)
    
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
        # Use BeautifulSoup to parse the HTML content
        soup = bs(driver.page_source, 'html.parser')
        # Serialize the DOM tree using BeautifulSoup's representation
        dom_tree = soup.prettify() #Make sure Dom is accesssibile as json 
        return dom_tree

    def createNode(element):
        pass #TODO 
    
    # Function to render next URL
    def retrieve():
        nonlocal urlIndex
        if (len(urls) > 0):
            
            url = urls[urlIndex]
            urlIndex += 1
              
            driver.get(url)
            time.sleep(2)  # Add some delay for page loading (adjust as needed)
              
              # Get paths
            image_path = getImagePath(urlIndex)
            pageId = getPageID(urlIndex)
            dom_tree_path = getDOMPath(urlIndex)
            html_path = getHTMLPath(urlIndex)
            list_path = getListPath()
            if(driver.current_url == url):
              # Get HTML and DOM content
              html_content = driver.page_source
              dom_tree = getDOMTree()

              # Get screenshot 
              driver.save_screenshot("screenshot.png")
              image = Image.open("screenshot.png")
              os.makedirs('data_news/images', exist_ok=True)
              image = image.convert("RGB")
              image.save(image_path, format="JPEG", quality=100) 
              callbackPerUrl("success", url, getPageID(urlIndex), dom_tree_path, html_path, list_path, dom_tree, html_content)         
            
            else:
              callbackPerUrl("failure", url, getPageID(urlIndex), dom_tree_path, html_path, list_path, None, None)
            # Continue with next URL
            retrieve()
        else:
            callbackFinal(driver)
    listPath = getListPath()
    if os.path.exists(listPath):
      os.remove(listPath)

    retrieve()

def callbackPerUrl(status, url, pageID, dom_tree_path, html_path, listPath, dom_tree, html):
    if status == "success":
      print(url)
      if not os.path.exists(listPath):
        with open(listPath, 'w') as f:
          f.write(url)
      else:
        with open(listPath, 'a') as f:
          f.write("\n" + pageID + "\t" + url)
      dom_content = json.dumps(dom_tree)
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
        input_path = "data_news/"+prefix
        output_path = "data_news"
    else: 
        print("Usage: python download_news.py TRAININGFILENAME")
        
    
    # Read URLs from file
    with open(input_path) as f:
        urls = f.read().splitlines()
        f.close()
        

    # Run rendering
    RenderUrlsToFile(urls, output_path, prefix, callbackPerUrl, callbackFinal)
