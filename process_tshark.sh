pcap_dir="svm_pcap"
tshark_dir="svm_ts"

files=$pcap_dir/*

for file in $files
do
inp="${file##*/}" #get basename of file
outp=`echo $inp | sed "s/.pcap/.ts/g"`
outp="$tshark_dir/$outp"

echo "Processing $file, output to $outp"

tshark -r $file | tr -s " " | cut -d' ' -f 2,4,8 > $outp

done
