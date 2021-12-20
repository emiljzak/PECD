import os
import subprocess
import re


my_jobs      = subprocess.check_output(["sq"])
jobs_analyze = re.search(r"*ANA*",str(my_jobs))

print("ANALYZE type jobs: " + str(jobs_analyze))