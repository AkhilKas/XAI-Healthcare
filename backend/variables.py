from pathlib import Path
import torch

TASK = 2
MODEL_SAVE_PATH = Path(f"trained_models/rnn_model_task{TASK}.pt")

DOFS = 18
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE, "Device used")

P_CORES = 6

chan_name = {
    0: 'HPosX',
    1:	'HPosY',
    2: 	'HPosZ',
    3:	'HRotX',
    4:	'HRotY',
    5:	'HRotZ',
    6:	'LPosX',
    7:	'LPosY',
    8:	'LPosZ',
    9:	'LRotX',
    10:	'LRotY',
    11:	'LRotZ',
    12:	'RPosX',
    13:	'RPosY',
    14:	'RPosZ',
    15:	'RRotX',
    16:	'RRotY',
    17: 'RRotZ'
}
