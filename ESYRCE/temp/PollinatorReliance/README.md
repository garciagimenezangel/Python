Calculation of evolution of pollinators' reliance of crops and evolution of intensification metrics:

1: run calculateEvolutionDemand.py

    Calculate demand = Area x Reliance in every ESYRCE block, every year

    Calculate most recent value of demand, oldest value, and slope: (final-initial)/(final year - initial year) 

2: run calculateEvolutionIntensification.py

3: Analyze values of slope -> Identify the areas with the largest changes

CRS must be 'EPSG:23028' for layer z28 and 'EPSG:23030' for z30

