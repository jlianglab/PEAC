# Learning Anatomically Consistent Embedding for Chest Radiography
This is official code for our **BMVC 2023 Oral paper**:  
[Learning Anatomically Consistent Embedding for Chest Radiography](https://papers.bmvc2023.org/0617.pdf)

We have introduced a new self-supervised learning (SSL) method called *PEAC (patch embedding of anatomical consistency)*. Compared with photographic images, medical images acquired with the same imaging protocol exhibit high consistency in anatomy. To exploit this anatomical consistency, we propose to learn global and local consistencies via stable grid-based matching, transfer pre-trained *PEAC* model to diverse downstream tasks. *PEAC* (1) achieves significantly better performance than the existing state-of-the-art fully-supervised and self-supervised methods, and (2) can effectively captures the anatomical structure consistency between patients of different genders and weights and between different views of the same patient which enhances the interpretability of our method for medical image analysis. 

![Image of framework](images/architecture.jpg)
