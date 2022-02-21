FILE=covid_ct_resnet_checkpoint.pth
if [ ! -f "$FILE" ]; then
    echo "Downloading $FILE ..."
    # ResNet34
    # https://drive.google.com/file/d/1PoSUIocSyi4jlFx4Dp7w3-7UcDSYA9I6/view?usp=sharing
    gdown --id 1PoSUIocSyi4jlFx4Dp7w3-7UcDSYA9I6
else
    echo "$FILE Exist!"
fi


FILE=covid_ct_mobile_checkpoint.pth
if [ ! -f "$FILE" ]; then
    echo "Downloading $FILE ..."
    # MobileNet_V2
    # https://drive.google.com/file/d/1PgoLOMhdmCE3HC8G_pNIPqPb4yMkQw0e/view?usp=sharing
    gdown --id 1PgoLOMhdmCE3HC8G_pNIPqPb4yMkQw0e
else
    echo "$FILE Exist!"
fi