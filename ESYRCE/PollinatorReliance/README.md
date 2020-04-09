Calculation of evolution of pollinators' reliance of crops and evolution of intensification metrics:

1: Filter data by spatial extent: run filterDataByExtent.py, for layers z28 and z30

2: run calculateEvolutionDemand.py

    Calculate demand = Area x Reliance in every ESYRCE block, every year

    Calculate most recent value of demand, oldest value, and slope: (final-initial)/(final year - initial year) 

3: rub calculateEvolutionIntensification.py

4: Analyze values of slope -> Identify the areas with the largest changes

CRS must be 'EPSG:32628' for layer z28 and 'EPSG:32630' for z30

