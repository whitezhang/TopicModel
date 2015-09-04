Before running the model, delete or save the model file. Otherwise, the content will be appended at the end of the previous contents.

# Small Test
## HINT: the file path needs changes (in data folder)

python new.py --newsf htm-news-14.txt -k 5 --alpha=0.5 --beta=0.5 --tweetsf glasgow-atm-content-14-4.txt -a glasgow-atm-others-14-4.txt -i 2

python lda.py -f htm-news-14.txt --stopwords True -k 20 --alpha=0.5 --beta=0.5 -i 1

python time_lda.py -f htm-news-14.txt --stopwords True -k 20 --alpha=0.5 --beta=0.5 -i 1

# Big file

python new.py --newsf 0715_story.txt -k 10 --alpha=0.5 --beta=0.5 --tweetsf glasgow-atm-content-1000.txt -a glasgow-atm-others-1000.txt —stopwords True -i 100

python lda.py -f lda_content.txt --stopwords True --alpha=0.5 --beta=0.5 -i 100 -k 20

python time_lda.py -f lda_content.txt --stopwords True --alpha=0.5 --beta=0.5 -i 100 -k 2

