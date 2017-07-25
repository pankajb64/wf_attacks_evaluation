pcap_dir='pcap'
all_sites='all_sites'
out_dir='svm_traces'
i=1
while read -r site
do
	j=1
	for file in $pcap_dir/*-$site*
	do
		#echo $file
		filename="${out_dir}/${i}_${j}.pcap"
		echo "Copying $file as $filename"
		cp $file $filename
		j=$((j+1))
	done
	#echo "$site - $j"		
	i=$((i+1))
done < $all_sites
