import os
from bs4 import BeautifulSoup
import requests
import re
import time
import json

outfile = open("/media/larry/mydiskishuge/code/autotainment/output.txt", "w")

output = ""

for i in os.listdir("/media/larry/mydiskishuge/code/autotainment/output/copypasta/"):
    j = open(f"/media/larry/mydiskishuge/code/autotainment/output/copypasta/{i}", "r")

    x = json.loads(str(j.read().lower()))

    output += re.sub(r"[^a-z0-9#$%&/. -]+", r"", x["selftext"])
    output += " "
    

output = re.sub(r"[ ]{2,}", r" ", output)
outfile.write(output)

outfile.close()