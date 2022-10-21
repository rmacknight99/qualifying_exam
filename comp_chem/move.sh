for i in */;
do
    cd $i
    mkdir orca/
    mv *opt_orca.xyz orca/
    cd ..
done
