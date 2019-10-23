awk -F\t '{print $1,"\t", $4}' compiled_books_content.txt | python filter.py > /tmp/tmp.txt
./doc2mat -skipnumeric /tmp/tmp.txt book_mat.cluto
# please download and install GKlib: https://github.com/KarypisLab/GKlib
csrcnv book_mat.cluto 1 book_mat.coo 6
cat book_mat.coo | python trim.py
