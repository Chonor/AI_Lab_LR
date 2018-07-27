
import csv
import glob

for filename in glob.glob(r'*.csv'):
    with open (filename,'r') as url:
        with open (filename[:-4]+'.txt','w')as process:
            process.write (url.read().replace(',','\t').replace('?','0'));
            