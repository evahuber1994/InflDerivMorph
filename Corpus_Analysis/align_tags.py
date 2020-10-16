

def morph_info(tag_str, splitter, pos):
    infs = dict()
    if pos == "ADJ":
        split_tag = tag_str.split(splitter)
        for t in split_tag:
            if "=" in t:
                t_split = t.split("=")
                type = t_split[0]
                value = t_split[1]
                if "," in value:
                    infs[type] = value.split(",")

                else:
                   infs[type] = [value]
    else:
        tag = tag_str[2:]
        infs = translate_into_ud(tag, pos)

    return infs

def check_dict(in_dict, tag, current, inf, l_s):
    if tag in in_dict:
        length_old = in_dict[tag][0]
        if l_s > length_old:
            inf.insert(0, l_s)
            in_dict[tag] = inf
            #in_dict[tag][1] = current
            #in_dict[tag][0] = l_s
        elif l_s == length_old:
            c = in_dict[tag][1]
            if type(c) == "int":
                in_dict[tag][1] = c + current
            else:
                in_dict[tag][1] = current
        else:
            return in_dict

    else:
        inf.insert(0, l_s)
        in_dict[tag] = inf


    return in_dict

def create_dict_ambiguous(path):
    dicty = dict()
    with open(path, 'r') as rf:
        next(rf)
        for l in rf:
            l = l.strip()
            line = l.split("\t")
            tw = line[0] +"*" +line[1]
            if tw not in dicty:
                dicty[tw] = 0
            dicty[tw] += int(line[2])
    return dicty

def to_file_ambiguous(dict_count,out_path):
    with open(out_path, 'w') as wf:
        wf.write("{}\t{}\t{}\n".format("relation", "word", "count"))
        for l,v in dict_count.items():
            k = l.split("*")
            wf.write("{}\t{}\t{}\n".format(k[0], k[1], str(v)))



def create_dict(path):
    dicty = dict()
    with open(path, 'r') as rf:
        next(rf)
        for l in rf:
            l = l.strip()
            line = l.split("\t")


            if len(line) > 4:
                tag_c= line[4]
            elif len(line) == 4:
                tag_c = "NA"
            elif len(line) == 3:
                lab = target_word + "_" + tag_ac
                inf = line
                check_dict(dicty, lab, counts, inf, 0)
                continue
            if "|" in tag_c:
                tag_c_tag = morph_info(tag_c, "|", "ADJ")
            else:
                tag_c_tag = [tag_c]
            tag_ac = line[0]
            pos_c = line[3]
            if line[2] != "NA":
                counts = int(line[2])
            else:
                counts = 0
            target_word = line[1]
            inf = [counts, target_word, tag_ac, pos_c, tag_c]
            if tag_ac.startswith("INF"): # word is an inflection

                #tag_ac = tag_ac.strip("INF_")
                tag_ac = tag_ac[4:]
                if "|" in tag_ac: # adjectives
                    pos_ac = "ADJ"
                    delimiter ="|"
                else:
                    delimiter = ";"
                    pos_ac = get_pos(tag_ac[0])
                if pos_c == pos_ac:
                    tag_ac_tag = morph_info(tag_ac, delimiter, pos_ac)
                    if tag_c == "NA":
                        lab = target_word + "_" + tag_ac
                        dicty = check_dict(dicty, lab, counts, inf, 0)
                    else:
                        shared = dict()
                        for k, v in tag_ac_tag.items():
                            if k in tag_c_tag:
                                tag = tag_c_tag[k]
                                if v[0] in tag:
                                    shared[k] = tag
                        length_shared = len(shared)
                        lab = target_word + "_" + tag_ac
                        dicty = check_dict(dicty, lab, counts, inf, length_shared)
            else: # word is a derivation
                tag_ac = tag_ac.strip("DER_")
                print(tag_ac)
                if tag_ac.startswith("d"):
                    pos = get_pos(tag_ac[2])
                else:
                    pos = get_pos(tag_ac[1])
                if pos == pos_c:
                    lab = target_word + "_" + tag_ac
                    dicty = check_dict(dicty, lab, counts, inf,0)

    return dicty

def to_file(dicty, out_path):
    with open(out_path, 'w') as wf:
        wf.write("{}\t{}\t{}\t{}\t{}\n".format("relation", "word", "frequency", "corpus_pos", "corpus_morph"))
        for k,v in dicty.items():
            tw = k.split("_")[0]
            rel = k.split("_")[1]
            print(v)
            if len(v) == 6:
                wf.write("{}\t{}\t{}\t{}\t{}\n".format(rel, tw, v[1], v[4], v[5])) #[counts, target_word, tag_ac, pos_c, tag_c]
            else:
                wf.write("{}\t{}\t{}\t{}\t{}\n".format(rel, tw, v[1], "NA", "NA"))
##helper methods

def get_pos(char):
    if char == "V":
        return "VERB"
    elif char =="N":
        return "NOUN"
    elif char =="A":
        return "ADJ"
    else:
        print("pos wrong char, {}".format(char))
def get_singplural(char):
    if char == "PL":
        return "Plur"
    elif char == "SG":
        return "Sing"
    else:
        print("number wrong char {}".format(char))
        return ""
def get_case(char):
    if char == "NOM":
        return "Nom"
    elif char =="ACC":
        return "Acc"
    elif char == "DAT":
        return "Dat"
    elif char == "GEN":
        return "Gen"
    else:
        print("case wrong char {}".format(char))
        return ""

def translate_into_ud(str_unimorph, pos):
    translation = dict()
    if pos == "NOUN":
        if ";" in str_unimorph:
            split_str = str_unimorph.split(";")
        else:
            print("no ; in string {}".format(str_unimorph))
            return ""
        translation["Case"] = get_case(split_str[0])
        translation["Number"] = get_singplural(split_str[1])
    elif pos == "VERB":
        if ";" in str_unimorph:
            split_str = str_unimorph.split(";")
        else:
            print("no ; in string {}".format(str_unimorph))
            return ""

        if split_str[0] == ".PTCP":
            translation["VerbForm"] = ["Part"]
        elif split_str[0] == "NFIN":
            translation["VerbForm"] = ["Inf"]
        else:
            if split_str[0] == "IND" or split_str[0] == "IMP" or split_str[0] == "SBJV":
                translation["VerbForm"] = ["Fin"]
                translation["Person"] = [split_str[-2]]
                translation["Number"] = [get_singplural(split_str[-1])]




    else:
        print("unknown pos: {}".format(pos))

    return translation

def main():
    in_path = "/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data/FINAL/DE/new_counts_german.csv"
    out_path = "/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data/FINAL/DE/new_counts_german_freqs_ambgs.csv"
    out_dict = create_dict_ambiguous(in_path)
    #print(out_dict)
    to_file_ambiguous(out_dict, out_path)
if __name__ =="__main__":
    main()