import sys
import os

base_path = "/Users/alxcoh/Documents/PythonEclipseWorkspace/QuestionAskerEverything"
all_paths = []

for root, subdirs, files in os.walk(base_path):
    if not '.git' in root:
    	all_paths.append(root)

for p in all_paths:
	sys.path.append(p)

base_path = base_path + "/"


