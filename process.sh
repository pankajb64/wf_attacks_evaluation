pcap_dir="mon_ts"
feat_dir="mon_feat"

files=$pcap_dir/*

for file in $files
do
inp="${file##*/}" #get basename of file
outp=`echo $inp | sed "s/.ts/.feat/g"`
outp="$feat_dir/$outp"

echo "Processing $file, output to $outp"
python process.py $file $outp

done
