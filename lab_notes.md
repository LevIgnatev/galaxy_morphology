## 2025-08-31
Added an improved, more detailed description of the project in the README (still to be improved). Planning on downloading the data tomorrow.

## 2025-09-01
Downloaded the galaxy zoo 2 dataset (~3 GB) and the Sloan Digital Sky Survey (DR18 and 17) (a total of ~1 GB)

## 2025-09-05
Decided to use the galaxy zoo 2 dataset. With the help of AI (results that this task is incredibly difficult due to the data format) over the last couple of days managed to save the mapped csv with data. 
A snippet was saved to data/labels. The manifest with labels consists of 92 columns representing features of every image, as well as the scores in each category (spiral, smooth, merger, etc.)
The last column corresponds to the final derived label, with a threshold of certainty of 0.6 (if not met - the galaxy is classified as ambiguous).

