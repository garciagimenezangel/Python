import ee
from ee_plugin import Map

basauri = ee.Geometry.Point(-2.89, 43.2367)

options = {
  title: 'Temperature series',
  vAxis: {title: 'Temperature (K)'},
  hAxis: {title: 'Year' },
  legend: {position: 'none'}
}

tempCollKdeg = ee.ImageCollection("ECMWF/ERA5/MONTHLY").filterDate('1980-01-01', '2020-01-01').select('mean_2m_air_temperature')

tempChart = ui.Chart.image.series({
  imageCollection: tempCollKdeg,
  region: basauri,
  xProperty: 'system:time_start'
})
tempChart.setOptions(options)

#
#addTime = function(image) {
#  return image.addBands(image.metadata('system:time_start')
#    .divide(1000 * 60 * 60 * 24 * 365))
#};
#tempCollTime = tempColl.map(addTime)
#
#// Reduce the collection with the linear fit reducer.
#// Independent variable are followed by dependent variables.
#linearFit = tempCollTime.select(['system:time_start', 'mean_2m_air_temperature'])
#  .reduce(ee.Reducer.linearFit())
#print(linearFit)
#
#Map.addLayer(linearFit.select("scale"))
#Map.centerObject(basauri,10)