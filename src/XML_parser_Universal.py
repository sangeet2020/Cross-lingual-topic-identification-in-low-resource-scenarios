#!/usr/bin/env python
# coding: utf-8

'''
Create parallel text from XM files
'''

#for eng run:
# python3 XML_parser_Universal.py <eng> <path for source data> <eng.ids> <eng.text>


#for target lang(hin, swa) run:
# python3 XML_parser_Universal.py <eng> <path for source data> <path for ids file of source data> <hin.ids> <hin.text>

import sys
import os
import codecs
from glob import glob
from collections import OrderedDict
import xml.etree.ElementTree as ET


# In[2]:

def langauge_Selection(lang_code):

    if lang_code == 'eng':
        source_language()
    else:
        target_language()


def source_language():

    eng_path = str(sys.argv[2])
    ids = str(sys.argv[3])
    text = str(sys.argv[4])
    #path = "/home/sangeet/ltf/ltf/hin/"
    xml_files = glob(eng_path + "*.xml")
    xml_files
    print_XML (ids, text, xml_files)

    # In[3]:


def target_language():

    path = str(sys.argv[2])
    src_lang_path = str(sys.argv[3])
    ids = str(sys.argv[4])
    text = str(sys.argv[5])
    #path = "/home/sangeet/ltf/ltf/hin/"
    xml_files = glob(path + "*.xml")

    #src_lang_path = "/home/sangeet/ltf/ltf/eng/English.ids"

    with open(src_lang_path) as f:
        lines = [line.rstrip('\n') for line in f]

    basename_list = [os.path.basename(line).split(".")[0] for line in lines]
    #for line in lines:
    #    basename = os.path.basename(line)
    #    basename_list.append(basename.split(".")[0])


    xml_files = [path + i + ".ltf.xml" for i in basename_list]
    print('xml_files:', len(xml_files))
    #for i in basename_list:
    #    xml_files.append(path + i + ".hin.ltf.xml")

    xml_files = list(OrderedDict.fromkeys(xml_files))
    print_XML (ids, text, xml_files)


def print_XML (ids, text, xml_files):

    #createFolder('./data/')
    f_doc_id = codecs.open(ids, "w", 'utf-8')
    f_org_text = codecs.open(text, "w", 'utf-8')

    for xml_file in xml_files:
        if not os.path.exists(xml_file):
            print(xml_file, 'NOT FOUND.')
            sys.exit()
        tree = ET.parse(xml_file)
        root = tree.getroot()
        #LCTL_TEXT

        for child in root:
            #DOC
            for gchild in child:
                #TEXT
                for ggchild in gchild:
                    #SEG
                    f_doc_id.write(child.attrib['id']+'.'+ggchild.attrib['id']+'\n')
                    for org_text in ggchild.findall('ORIGINAL_TEXT'):
                        #ORIGINAL_TEXT
                        text = org_text.text
                        f_org_text.write(text+'\n')

    f_doc_id.close()
    f_org_text.close()

# def createFolder(directory):
#     try:
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#     except OSError:
#         print ('Error: Creating directory. ' +  directory)


if 5 <= len(sys.argv) <= 6:

    LANG_CODE = str(sys.argv[1])
    langauge_Selection(LANG_CODE)

else:

    print(len(sys.argv))

    print("python3 XML_parser_Universal.py <eng> <path for source data> <eng.ids> <eng.text>")

    print("python3 XML_parser_Universal.py <il> <path for source data>",
          "<path for ids file of source data> <il.ids> <il.text>")

    sys.exit()
