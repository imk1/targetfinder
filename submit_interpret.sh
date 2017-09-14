cd /home/brianhie/bqtl/targetfinder/

cat ./tfs.txt | \
    while read T
    do
        qsub -N $T -v TF=$T,METRIC="recursive" interpret.sh
        break
#        TF=$T METRIC="oob" sh interpret.sh &
    done
