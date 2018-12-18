import re
from fileio import FileIO
from preprocess import Clean
from nltk import word_tokenize

io = FileIO()
clean = Clean()


def read_IntergrateCons(path):
    data = io.read_file_text(path).split("\n")
    data_clean = []
    for i in data:
        i_clean = re.sub("<Cons>", '', i)
        i_clean = re.sub("</Cons>", '', i_clean)
        data_clean.append(i_clean)
    io.write_file_text("\n".join(data_clean), "data/test_cons")

def read_IntergratePros(path):
    data = io.read_file_text(path).split("\n")
    data_clean = []
    for i in data:
        i_clean = re.sub("<Pros>", '', i)
        i_clean = re.sub("</Pros>", '', i_clean)
        data_clean.append(i_clean)
    io.write_file_text("\n".join(data_clean), "data/test_pros")


data_pros  = io.read_file_text("data/test_pros_clean").split("\n")
data_cons  = io.read_file_text("data/test_cons").split("\n")

# data_pros_clean = []
# data_cons_clean = []
# for rev in data_pros:
#     doc_clean = clean.clean_review(rev)
#     doc_clean_process_number = clean.number_process(doc_clean)
#     data_pros_clean.append(doc_clean_process_number)
#     io.write_file_text("\n".join(data_pros_clean), "data/test_pros_clean")
#
#
# for rev in data_cons:
#     doc_clean = clean.clean_review(rev)
#     doc_clean_process_number = clean.number_process(doc_clean)
#     data_cons_clean.append(doc_clean_process_number)
#     io.write_file_text("\n".join(data_cons_clean), "data/test_cons_clean")

for i in data_pros[:4]:
    print(word_tokenize(i))