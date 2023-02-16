#!/usr/bin/env bash

for j in {'astemizole','azimilide','bepridil','chlorpromazine','cisapride','clarithromycin','clozapine','diltiazem','disopyramide','dofetilide','domperidone','droperidol','ibutilide','loratadine','metoprolol','mexiletine','nifedipine','nitrendipine','ondansetron','pimozide','quinidine','ranolazine','risperidone','sotalol','tamoxifen','terfenadine','vandetanib','verapamil'}
do
	python save_FDA_data.py --drug $j
	python save_FDA_data_all.py --drug $j
done
