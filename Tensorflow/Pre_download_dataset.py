import os
import zipfile 
import wget #pip install wget

url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path='./tmp'
file_name=path+'/cats_and_dogs_filtered'
local_zip = file_name+'.zip'


#check(create) the path
try: 
    os.mkdir(path)
except:
    pass

#download file
if not os.path.exists(file_name+'.zip'):
    print('Beginning download cat and dog images for training')
    wget.download(url, file_name+'.zip')


# Unzip the archive
print('unziping')
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(path)
zip_ref.close()
print('finished unzip')
