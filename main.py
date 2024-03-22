from cli.cli import CustomLightningCLI
import warnings
from models import base
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Gets rid of warning about num. of workers
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*shuffle*")

def cli_main():
    # Models used MUST inherit from base.BaseModel
    CustomLightningCLI(base.BaseModel, subclass_mode_model=True)

if __name__ == "__main__":
    cli_main()