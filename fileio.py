import json
import os


class FileIO():

    def __init__(self):
        print("")

    def read_file_json(self, path_file):
        with open(path_file, 'r') as content:
            data = json.load(content)
        content.close()
        return data

    def read_multi_files_json(self,path_dir):
        data = []
        listfiles = os.listdir(path_dir)
        for file_name in listfiles:
            for ele in self.read_file_json(path_dir + "/" + file_name):
                data.append(ele)
        return data

    def read_file_text(self, path_file):
        with open(path_file, 'r') as content:
            data = content.read()
        content.close()
        return data

    def read_file_text_by_lines(self,path_file):
        with open(path_file, 'r') as content:
            data = content.readlines()
        content.close()
        return data

    def write_file_json(self, data, path_file):
        outfile = open(path_file, 'w')
        json.dump(data, outfile)
        outfile.close()

    def write_file_text(self, data, path_file):
        out = open(path_file, 'w')
        out.write(data)
        out.close()

    def write_tuple_data(self, data, path_write):
        corpus = []
        for label, revi in data:
            corpus.append(str(label) + "--" + revi)
        self.write_file_text('\n'.join(corpus), path_write)
