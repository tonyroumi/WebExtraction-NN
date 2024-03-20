# Neural network for Web Extraction
This project aims to harness the power of neural networks to facilitate web extraction tasks, making the process more robust and efficient.

This was inspired by Tomas Gogar(B), Ondrej Hubacek, and Jan Sedivy, reimplemented for article extraction.
https://link.springer.com/content/pdf/10.1007/978-3-319-44944-9_14.pdf

## Overview
In the realm of web extraction, web scrapers are custom scripts built for a particular website. They rely heavily on the structure of the web page and patterns within the HTML to be able to dynamically scrape desired information. In this project I present a method to extract information from any web page using a deep convolutional network. I utilize images of web pages and information from their corresponding HTML documents to train a network to identify class elements from new, unseen pages. I use beautiful soup and selenium to obtiain dataset. In this project I combine textual data with visual data and methods for computer vision to create a robust, dynamic web scraper.

