Расмотрим способ сжатия изображения, основанный на SVD разложении. Для самого SVD рассмотрим три реализации \
1) numpy_svd - svd из numpy

2) simple_svd - power iterations method

3) advanced_svd - block power iterations method

Для каждого из способов рассмотрим сжатие исходной картинки в 10, 50 и  100 раз

original

![alt text](images/origin_lena.bmp)

numpy_svd

| * | 10    | 50    | 100 |
| :---:   | :---: | :---: | :---: |
| numpy_svd |  ![alt text](images/numpy_10_new_lena.bmp)  | ![alt text](images/numpy_50_new_lena.bmp)   | ![alt text](images/numpy_100_new_lena.bmp) |
| simple_svd |  ![alt text](images/simple_10_new_lena.bmp)  | ![alt text](images/simple_50_new_lena.bmp)   | ![alt text](images/simple_50_new_lena.bmp) |
| advanced_svd |  ![alt text](images/advanced_10_new_lena.bmp)  | ![alt text](images/advanced_50_new_lena.bmp)   | ![alt text](images/advanced_100_new_lena.bmp) |