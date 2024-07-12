cd /home/maith/Desktop/Cityscapes/gtFine_trainvaltest/train/aachen

for img in $(ls /home/maith/Desktop/Cityscapes/leftImg8bit_trainvaltest/train/aachen/*.png); do
    base=$(basename $img "_leftImg8bit.png")
    
    if [ ! -f "${base}_gtFine_color.png" ] || \
       [ ! -f "${base}_gtFine_instanceIds.png" ] || \
       [ ! -f "${base}_gtFine_labelIds.png" ] || \
       [ ! -f "${base}_gtFine_polygons.json" ]; then
       echo "Missing annotation files for $img"
    else
       echo "All annotation files match for ${base}"
    fi
done
