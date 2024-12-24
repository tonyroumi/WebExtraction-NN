# Neural network for Web Extraction
This project utilizes a neural network to facilitate web extraction tasks.

This was inspired by Tomas Gogar(B), Ondrej Hubacek, and Jan Sedivy, reimplemented in PyTorchfor article extraction.
https://link.springer.com/content/pdf/10.1007/978-3-319-44944-9_14.pdf

## Overview
Web scrapers rely heavily on the structure of the web page and patterns within the HTML to be able to dynamically scrape desired information. In this project I present a method to extract information from any web page using a convolutional network. I utilize images of web pages and information from their corresponding HTML documents to train a network to identify class elements from new, unseen pages. I use beautiful soup and selenium to obtiain dataset. In this project I combine textual data with visual data and methods for computer vision to create a robust, dynamic web scraper.

