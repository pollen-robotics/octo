import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from octo.model.octo_model import OctoModel

model = OctoModel.load_pretrained("/data1/apirrone/octo/trainings/")

task = model.create_tasks(texts=["Grab the wooden cube"])

