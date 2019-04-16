#!/bin/bash
baseDir=/Users/wangmingjian/All_code/python/YAYA/data_in/pic/blurred_coil-100_ver1.1
targetDir=/Users/wangmingjian/All_code/python/YAYA/data_in/pic/batch_blurred
folderCount=3

echo "The base directory is: $baseDir"
echo "The target directory is: $targetDir"

[ -f "${0}.lock" ] && echo "${0}.lock locked, maybe ${0} processing now!" \
&& exit -1

[ ! -d $baseDir ] && echo "warning: baseDir=$baseDir is not a directoy, exit!" && exit

lockfile ${0}.lock

for((i=1;i<=folderCount;i++))
do
	[ ! -d $targetDir$i ] && mkdir $targetDir$i
done

fileCount=`find $baseDir -type f | wc -l`
if [ $fileCount = 0 ] 
then
	echo "baseDir $baseDir have $fileCount files"
	rm -f ${0}.lock
	exit
fi


x=0
for fileName in $baseDir/*;
do
	fileList[$x]=$fileName
	let "x+=1"
done

total=${#fileList[@]}
echo "get files total: $total"
avg=`expr $(($total + $folderCount - 1)) / $folderCount`

for((i=0,p=0;i<${#fileList[@]};i++))
do
	if [ `expr $i % $avg` = "0" ]
	then
		let "p+=1"
	fi
	\cp ${fileList[$i]} $targetDir$p
done

#
#for((i=0;i<${#fileList[@]};i++))
#do
#	\rm ${fileList[$i]}
#done


rm -f ${0}.lock
echo "success end!"
