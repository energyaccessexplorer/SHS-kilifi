var aoi = /* color: #d63000 */ee.Geometry.Polygon(
        [[[36.86563232197536, -1.2987724874475357],
          [36.86563232197536, -1.3118153666772525],
          [36.884515073440205, -1.3118153666772525],
          [36.884515073440205, -1.2987724874475357]]], null, false);

// Function to mask clouds in Sentinel-2 data
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000);
}

// Load Sentinel-2 MSI data
var dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(aoi)
                  .filterDate('2023-01-01', '2023-12-31')
                  // Pre-filter to get less cloudy granules.
                  //.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                  .map(maskS2clouds);

var visualization = {
  min: 0.0,
  max: 0.3,
  bands: ['B4', 'B3', 'B2'],
};
//Map.addLayer(dataset.mean(), visualization, 'RGB');

// Load Sentinel-1 SAR data
var sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)
  .filterDate('2023-01-01', '2023-12-31')
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('instrumentMode', 'IW'));

// Calculate VH and VH_savg
var vh = sentinel1.select('VH').median().rename('VH');
var vh_savg = sentinel1.select('VH').mean().rename('VH_savg');

// Calculate indices: NDWI, NDBI, NDVI
var addIndices = function(image) {
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI');
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndwi).addBands(ndbi).addBands(ndvi);
};

var sentinel2_with_indices = dataset.map(addIndices);

// Annual median composite
var annual_composite = sentinel2_with_indices.median()
  .addBands(vh)
  .addBands(vh_savg);

//Map.addLayer(annual_composite, {}, 'Annual Composite');

// Load solar polygons
var solarPolygons = ee.FeatureCollection('projects/ee-chancy/assets/Solar_Panel_Training_Data')
  .filter(ee.Filter.eq('class', 1));
print('Solar Polygons:', solarPolygons);

// Generate negative samples
var nonSolarArea = aoi.difference(solarPolygons.geometry(), ee.ErrorMargin(1));
print('Non-Solar Area:', nonSolarArea);

// Create points for training: inside solar panel polygons (class 1) and outside (class 0)
var numNonSolarSamples = 1500;

// Sample solar panel polygons directly
var solarSamples = solarPolygons.map(function(feature) {
  return feature.set('class', 1);
});
print('Solar Samples:', solarSamples);

// Generate random points within non-solar areas
var nonSolarPoints = ee.FeatureCollection.randomPoints(nonSolarArea, numNonSolarSamples, 50)
  .map(function(feature) {
    return feature.set('class', 0);
  });
print('Non-Solar Points:', nonSolarPoints);

// Combine the samples
var trainingSamples = solarSamples.merge(nonSolarPoints);
print('Training Samples:', trainingSamples);

// Split training data into training and validation datasets
var trainingSplit = 0.7;
var withRandom = trainingSamples.randomColumn('random');
var trainingData = withRandom.filter(ee.Filter.lt('random', trainingSplit));
var validationData = withRandom.filter(ee.Filter.gte('random', trainingSplit));
print('Training Data:', trainingData);
print('Validation Data:', validationData);

// Visualize training and validation data
//Map.addLayer(trainingData, {color: 'red'}, 'Training Data');
//Map.addLayer(validationData, {color: 'blue'}, 'Validation Data');

// Sample the training data from the annual composite image
var training = annual_composite.sampleRegions({
  collection: trainingData,
  properties: ['class'],
  scale: 10
});
print('Training Sample:', training);

// Train a Decision Tree classifier
var classifier = ee.Classifier.smileCart(10).train({
  features: training,
  classProperty: 'class',
  inputProperties: annual_composite.bandNames()
});

// Classify the image
var classified = annual_composite.classify(classifier);

// Sample the validation data from the annual composite image
var validation = annual_composite.sampleRegions({
  collection: validationData,
  properties: ['class'],
  scale: 10
});

// Evaluate the classifier accuracy
var validated = validation.classify(classifier);
var confusionMatrix = validated.errorMatrix('class', 'classification');
var accuracy = confusionMatrix.accuracy();
var producersAccuracy = confusionMatrix.producersAccuracy();
var usersAccuracy = confusionMatrix.consumersAccuracy();

print('Confusion Matrix:', confusionMatrix);
print('Overall Accuracy (%):', accuracy.multiply(100));
print('Producers Accuracy (%):', producersAccuracy.multiply(100));
print('Users Accuracy (%):', usersAccuracy.multiply(100));

// Display the resultant map
Map.centerObject(aoi, 12);
Map.addLayer(classified, {min: 0, max: 1, palette: ['white', 'green']}, 'Solar Panels Detection');

// Ensure the classified image has only one band for vectorization
var classifiedSingleBand = classified.select('classification');

// Clip the classified image to the AOI
var classifiedClipped = classifiedSingleBand.clip(aoi);

// Convert the classified image to binary for solar panel detection
var solarPanelBinary = classifiedClipped.eq(1).rename('solar_panel');

// Convert the binary image to vector polygons
var solarPanelVectors = solarPanelBinary.reduceToVectors({
  geometryType: 'polygon',
  reducer: ee.Reducer.countEvery(),
  scale: 10,
  maxPixels: 1e9,
  geometry: aoi
});

// Display the vector polygons on the map
Map.addLayer(solarPanelVectors, {color: 'red'}, 'Solar Panels Polygons');

// Export the solar panel polygons to a shapefile or GeoJSON
Export.table.toDrive({
  collection: solarPanelVectors,
  description: 'Solar_Panels_Polygons',
  fileFormat: 'SHP'
});