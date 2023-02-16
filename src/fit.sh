#!/usr/bin/env bash

k="li"
i="2i"
for j in {'astemizole','azimilide','bepridil','chlorpromazine','cisapride','clarithromycin','clozapine','diltiazem','disopyramide','dofetilide','domperidone','droperidol','ibutilide','loratadine','metoprolol','mexiletine','nifedipine','nitrendipine','ondansetron','pimozide','quinidine','ranolazine','risperidone','sotalol','tamoxifen','terfenadine','vandetanib','verapamil'}
do
	python -u fit.py --model $i --drug $j --base_model $k --repeats 10 --verbose
done
