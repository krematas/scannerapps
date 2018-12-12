## Path setting
```
export PATH=/home/krematas/code/scanner/thirdparty/install/bin:$PATH
export LD_LIBRARY_PATH=/home/krematas/code/scanner/thirdparty/install/lib:$LD_LIBRARY_PATH
source .virtualenvs/soccer/bin/activate

```



## Instance segmentation (requires CPUs only)
```
python segment_cpp.py --path_to_data ~/data/barcelona/
```


## Depth estimation (requires GPUs)
```
python estimate_depth.py --path_to_data ~/data/barcelona/ --path_to_model ~/data/model.pth
```