import os
import sys
import webbrowser
from shiny._main import main

"""
File used for compiling the .exe version of the software.
"""


path = os.path.dirname(os.path.abspath(__file__))
apath = os.path.join(path, "app.py")

# these next two lines are only if you are using Windows OS
drive, apath = os.path.splitdrive(apath)
apath = apath.replace("\\","/")
# 

sys.argv = ['shiny', 'run', apath]
main()
# webbrowser.open("http://127.0.0.1:8000", new=2)
