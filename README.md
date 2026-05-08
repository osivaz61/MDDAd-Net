<img width="5619" height="4063" alt="arch" src="https://github.com/user-attachments/assets/92de74dc-1afa-4cc9-950d-0730e1f0b1fd" />


You can find the results (predicted masks, labels, and a Python file that calculates the metrics) on the following link: https://drive.google.com/file/d/1DYY3-4JlY_LYg89dDf3N6eMzwD2LnDhy/view?usp=drive_link <br>
The pre-trained models will be uploaded. -> OK <br>
(https://drive.google.com/drive/folders/11wYiAFXJQlStPh2gC5fkl41iMNRXKVIp?usp=sharing) <br>
Better models on the DR segmentation problem will be released. -> OK (shuntedSeg.py) <br>
The final code will be released. -> OK <br>

## Important
The results in the .out files are based on the coefficients in the .py files. The coefficients (lambda's) should be adjusted for these files that use the shuntedSeg model. They are arbitrary (they are for mambaVisionSeg). <br>

## Better ideas
For the IDRiD segmentation dataset: You can use the ddr grade part instead of the IDRiD grade part for the IDRiD segmentation dataset. Probably gives better results. <br>
Use the Shunted transformer Tiny model as a discriminator instead of the MambaVision Tiny model: it includes LayerNorm instead of BatchNorm. Probably gives better results. <br>

## Acknowledgements

This project is based on or inspired by the following repositories:

- [MambaVision](https://github.com/nvlabs/mambavision)
- [ShuntedTransformer](https://github.com/OliverRensu/Shunted-Transformer)
- [Log-vmamba](https://github.com/imedslab/LoG-VMamba)
