import errno
import glob
import os

import pandas as pd


def ner_concatenate(path):

    if not os.path.isdir(path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), path)

    lines = []
    sentence_id = 0
    for filename in glob.glob(os.path.join(path, "*.locs")):
        with open(filename, 'rt') as fd:
            for line in fd:
                line = line.replace("\n", "")
                line = line.replace(" ", "")
                line = line.replace("\x0c", "")
                line = line.replace("\xa0", "")
                line = line.replace("\u2009", "")
                line = line.replace("\u2002", "")
                line = line.replace("\u2003", "")
                line = line.replace("\u200a", "")
                processed_line = line.split("\t")
                if len(processed_line) > 1:
                    processed_line.append(str(sentence_id))
                    if len(processed_line) == 9 and len(processed_line[0]) > 0:
                        lines.append(processed_line)
                else:
                    sentence_id = sentence_id + 1


    df = pd.DataFrame.from_records(lines,
                                   columns=["token", "txt_source_file", "start_char", "end_char", "tag", "tag_id",
                                            "root_id", "relation", "sentence_id"])

    return df

