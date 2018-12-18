import re
from fileio import FileIO
from preprocess import Clean

io = FileIO()
clean = Clean()


## read cons, pros corpus from bing liu to add to train data

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



