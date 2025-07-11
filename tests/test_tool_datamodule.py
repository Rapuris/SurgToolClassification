import pytest
import torch
from pathlib import Path
from src.data.tool_datamodule import ToolDataModule
import logging

log = logging.getLogger(__name__)

@pytest.mark.parametrize("batch_size", [8, 32])
def test_tool_datamodule(tmp_path, batch_size):
    """
    Tests ToolDataModule to verify that it can be instantiated, setup, and that dataloaders
    return batches of the correct shape and type.
    Assumes at least one valid YOLOv8-format dataset is available at data_dirs[0].
    """
    # You may want to change this to a real data path for integration testing
    data_dirs = ["/mnt/data2/Sampath/RoboFlowDatasets/Sampath_Annotations/YOLO_finetune"]
    dm = ToolDataModule(data_dirs=data_dirs, batch_size=batch_size)
    dm.prepare_data()
    dm.setup()
    assert dm.data_train is not None
    assert dm.data_val is not None
    assert dm.train_dataloader() is not None
    assert dm.val_dataloader() is not None
    
    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    log.error(num_datapoints)
    log.error("train: " + str(len(dm.data_train)))
    log.error("val: " + str(len(dm.data_val)))
    log.error("test: " + str(len(dm.data_test)))

    # Check a batch from the train dataloader
    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    log.error(x.shape)
    log.error(y.shape)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64 or y.dtype == torch.long
    assert x.shape[0] == batch_size
    assert y.shape[0] == batch_size
    # Check image shape (C, H, W)
    assert x.ndim == 4
    assert x.shape[1] == 3  # RGB
    assert x.shape[2] == dm.image_size
    assert x.shape[3] == dm.image_size 
    
    
    
    
 