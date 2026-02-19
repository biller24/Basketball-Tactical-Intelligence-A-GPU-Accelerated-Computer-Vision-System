import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=api_key)
project = rf.workspace("workspace-5ujvu").project("basketball-players-fy4c2-vfsuv")
version = project.version(17)
dataset = version.download("yolov5")


new_project = rf.workspace("fyp-3bwmg").project("reloc2-den7l")
new_version = new_project.version(1)
new_dataset = new_version.download("yolov8")

