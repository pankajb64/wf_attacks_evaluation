#!/bin/bash
crawl_path="/media/sf_Documents/docker/results"
pcap_dir="pcap"

for i in {1..99..1}
do
	echo "Processing for site $i"
	for file in $crawl_path/crawl17061*/*/$i-*/*/*.pcap
	do
		#echo "Processing $file"
		tmp="${file##${crawl_path}/}"
		crawl_name="${tmp%%/*/*}"
		echo "crawl is $crawl_name"
		file_name="${file##*/}"
		temp="${file%/*/*}"
		site_name="${temp##*/}"
		
		cp_name="${pcap_dir}/${crawl_name}"
		cp_name="${cp_name}_${site_name}"
		cp_name="${cp_name}_${file_name}"
		
		echo "Processing $file, copying it to $cp_name"
		
		cp $file $cp_name
	done
	
done
