#!/usr/bin/env bash

for j in {'astemizole','azimilide','bepridil','chlorpromazine','cisapride','clarithromycin','clozapine','diltiazem','disopyramide','dofetilide','domperidone','droperidol','ibutilide','loratadine','metoprolol','mexiletine','nifedipine','nitrendipine','ondansetron','pimozide','quinidine','ranolazine','risperidone','sotalol','tamoxifen','terfenadine','vandetanib','verapamil'}
do
	python plot_FDA_data.py --drug $j
	python plot_FDA_data_all.py --drug $j
	python plot_FDA_data_saturation.py --drug $j
done
