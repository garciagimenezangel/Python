Calculation of evolution of pollinators' reliance of crops:

1: Filter data by spatial extent: run filterDataByExtent.py, for layers z28 and z30

2&3: run calculateEvolutionDemand.py

    2: Calculate demand = Area x Reliance in every ESYRCE block, every year

    3: Calculate most recent value of demand, oldest value, and slope: (final-initial)/(final year - initial year) 

4: Analyze values of slope -> Identify the areas with the largest changes in pollination demand

CRS must be 'EPSG:32628' for layer z28 and 'EPSG:32630' for z30

