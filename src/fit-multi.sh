#!/usr/bin/env bash

k="li"
#for i in {'1','2','2i','3','4','5','5i','6'}
#for i in {'7','8','9','10','11','12','13'}

for i in {'7','8','9','10','11','12','13'}
do
	for j in {'astemizole','azimilide','bepridil','chlorpromazine','cisapride','clarithromycin','clozapine','diltiazem','disopyramide','dofetilide','domperidone','droperidol','ibutilide','loratadine','metoprolol','mexiletine','nifedipine','nitrendipine','ondansetron','pimozide','quinidine','ranolazine','risperidone','sotalol','tamoxifen','terfenadine','vandetanib','verapamil'}
	do
		python -u fit.py --model $i --drug $j --base_model $k --repeats 10 --verbose #--fix_hill
	done
done
