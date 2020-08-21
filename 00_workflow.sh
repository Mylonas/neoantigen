#!/bin/bash

sampleIDs=("SRR057629" "SRR057630" "SRR057631" "SRR057632" "SRR057633" "SRR057634" "SRR057635" "SRR057636" "SRR057637" "SRR057638" "SRR057639" "SRR057640" "SRR057641" "SRR057642" "SRR057643" "SRR057644" "SRR057645" "SRR057646" "SRR057647" "SRR057648" "SRR057649" "SRR057650" "SRR057651" "SRR057652" "SRR057653" "SRR057654" "SRR057655" "SRR057656" "SRR057657" "SRR057658")    

genomeIndex="Homo_sapiens.GRCh37.dna_sm.primary_assembly"
gtfIndex="Homo_sapiens.GRCh37.87"  

userProject="scw1557"
mem="200G"
nodes="20"
runTime="69:05:00"

myDir=$(pwd)
scriptBase="00"
jobName="wflow"


for sampleID in "${sampleIDs[@]}"
do
	scriptName=${myDir}/temp/${scriptBase}.${sampleID}.sh
	rm -rf ${scriptName} || true
	touch ${scriptName}

	echo "#!/bin/bash" >> ${scriptName} 
        echo "#SBATCH -p c_compute_wgp" >> ${scriptName}
        echo "#SBATCH --mem=${mem}" >> ${scriptName}
        echo "#SBATCH --ntasks=${nodes}" >> ${scriptName}
        echo "#SBATCH --tasks-per-node=${nodes}" >> ${scriptName}
        echo "#SBATCH -t ${runTime}" >> ${scriptName}
        echo "#SBATCH -o ${myDir}/OUT/${scriptBase}${jobName}.%J" >> ${scriptName}
        echo "#SBATCH -e ${myDir}/ERR/${scriptBase}${jobName}.%J" >> ${scriptName}
        echo "#SBATCH --job-name=${jobName}" >> ${scriptName}
        echo "#SBATCH --account=${userProject}" >> ${scriptName}

	echo "module load java/1.8" >> ${scriptName}
	echo "module load raven" >> ${scriptName} 
	echo "module load STAR/2.5.1b" >> ${scriptName}	
	echo "module load tophat/2.0.11" >> ${scriptName} 
	echo "module load FastQC/0.11.8" >> ${scriptName}
	echo "module load samtools" >> ${scriptName}	

	echo "mkdir -p ${myDir}/../output/${sampleID}/fastqc/raw/" >> ${scriptName}	
	echo "mkdir -p ${myDir}/../output/${sampleID}/fastqc/trimmed/" >> ${scriptName}	
	echo "mkdir -p ${myDir}/../output/${sampleID}/tophat/" >> ${scriptName}

	## run fastqc on the raw fastq

	#echo "fastqc -o ${myDir}/../output/${sampleID}/fastqc/raw/ ${myDir}/../input/${sampleID}_1.fastq.gz ${myDir}/../input/${sampleID}_2.fastq.gz" >> ${scriptName}	

	## run the tophat mapping command
	#echo "tophat2 --transcriptome-index ${myDir}/../resources/${gtfIndex} -o ${myDir}/../output/${sampleID}/tophat ${myDir}/../resources/${genomeIndex} ${myDir}/../input/${sampleID}_1.fastq.gz ${myDir}/../input/${sampleID}_2.fastq.gz" >> ${scriptName} 

	## Data pre processing with opossum and variant calling with platypus
	echo "module purge" >> ${scriptName}
	echo "module load python-h5py/2.8.0" >> ${scriptName}
	echo "module load platypus" >> ${scriptName}
	echo "module load samtools" >> ${scriptName}
	echo "module load pip" >> ${scriptName}
	echo "pip install --user pysam==0.10.0 more-itertools argparse os-win runner" >> ${scriptName}  	

	echo "mkdir -p ${myDir}/../output/${sampleID}/opossum/" >> ${scriptName}
	echo "mkdir -p ${myDir}/../output/${sampleID}/platypus/" >> ${scriptName}

	#echo "python Opossum.py --BamFile=${myDir}/../output/${sampleID}/tophat/accepted_hits.bam --SoftClipsExist=False --ProperlyPaired=True --OutFile=${myDir}/../output/${sampleID}/opossum/${sampleID}.bam" >> ${scriptName}

	echo "module purge" >> ${scriptName}
	echo "module load python" >> ${scriptName}
	echo "module load platypus" >> ${scriptName}
	echo "module load pip" >> ${scriptName}

	#echo "python /apps/genomics/platypus/0.8.1.1/platypus-0.8.1.1/platypus-git/bin/Platypus.py callVariants --nCPU 20 --bamFiles ${myDir}/../output/${sampleID}/opossum/${sampleID}.bam --refFile ${myDir}/../resources/${genomeIndex}.fa --filterDuplicates 0 --minMapQual 0 --minFlank 0 --maxReadLength 500 --minGoodQualBases 10 --minBaseQual 20 --output ${myDir}/../output/${sampleID}/platypus/${sampleID}.vcf" >> ${scriptName}

	## run snpEFF
	echo "module load java" >> ${scriptName}
	echo "mkdir -p ${myDir}/../output/${sampleID}/snpEff/" >> ${scriptName}
	#echo "java -Xmx4g -jar snpEff/snpEff.jar GRCh37.75 ${myDir}/../output/${sampleID}/platypus/${sampleID}.vcf > ${myDir}/../output/${sampleID}/snpEff/${sampleID}.vcf" >> ${scriptName}

	##run fastaAlternator
	echo "module load GATK" >> ${scriptName}  
	echo "mkdir -p ${myDir}/../output/${sampleID}/gatk/" >> ${scriptName}     
	#echo "gatk IndexFeatureFile -F ${myDir}/../output/${sampleID}/platypus/${sampleID}.vcf" >> ${scriptName}
	#echo "gatk FastaAlternateReferenceMaker -R ${myDir}/../resources/${genomeIndex}.fa -O ${myDir}/../output/${sampleID}/gatk/${sampleID}.fasta -V ${myDir}/../output/${sampleID}/platypus/${sampleID}.vcf" >> ${scriptName} 	

	## run getfasta to create a chr delited fasta file
	#echo "mkdir -p ${myDir}/../output/${sampleID}/getfasta/" >> ${scriptName}
	#echo "module purge" >> ${scriptName}    
	#echo "module load compiler/gnu/7" >> ${scriptName} 
	#echo "module load mpi/intel/2018/2" >> ${scriptName} 
	#echo "module load hdf5" >> ${scriptName} 
	#echo "module load bedtools" >> ${scriptName} 
	#echo "bedtools getfasta -fi ${myDir}/../output/${sampleID}/gatk/${sampleID}.fasta -bed ${myDir}/../resources/${gtfIndex}.gtf -fo ${myDir}/../output/${sampleID}/getfasta/${sampleID}.fasta" >> ${scriptName}  
	
	## run netchop
	echo "mkdir -p ${myDir}/../output/${sampleID}/netchop/" >> ${scriptName}
	echo "netchop-3.1/netchop ${myDir}/../output/${sampleID}/getfasta/${sampleID}.fasta > ${myDir}/../output/${sampleID}/netchop/${sampleID}.out" >> ${scriptName}  
	#echo "netchop-3.1/netchop  ${myDir}/../output/${sampleID}/gatk/${sampleID}.fasta >  ${myDir}/../output/${sampleID}/netchop/${sampleID}.out" >> ${scriptName}

	chmod u+x ${scriptName}

	sbatch ${scriptName}

done


