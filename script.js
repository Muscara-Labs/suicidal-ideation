console.log('hello tensorflow');
let test_data;

function csvToJson(csvFile, callback) {
  const reader = new FileReader();

  reader.onload = (event) => {
    const csvData = event.target.result;
    const lines = csvData.split('\n');
    const headers = lines[0].split(',');

    const jsonData = [];

    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].split(',');
      if (line.length === headers.length) {
        const item = {};
        for (let j = 0; j < headers.length; j++) {
          if(parseInt(line[j]) !== undefined) {
            item[headers[j].trim()] = parseInt(line[j]);
          } else {
            item[headers[j].trim()] = line[j].trim();
          }
        }
        jsonData.push(item);
      }
    }

    if (typeof callback === 'function') {
      callback(jsonData);
    }
  };

  reader.readAsText(csvFile);
}

function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  const [xs, preds] = tf.tidy(() => {
    const xsNorm = tf.linspace(0, 1, 100);
    const predictions = model.predict(xsNorm.reshape([100, 1]));

    const unNormXs = xsNorm.mul(inputMax.sub(inputMin)).add(inputMin);
    const unNormPreds = predictions.mul(labelMax.sub(labelMin)).add(labelMin);

    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });


  const predictedPoints = Array.from(xs).map((val, i) => {
    return { index: val, value: preds[i] }
  });

  tfvis.render.barchart(
    { name: 'Model Predictions' },
    predictedPoints
  );
}

async function run(data) {
  const values = data.map(d => ({
    index: d.Index,
    value: d.Suicidal_label
  }));

  tfvis.render.barchart(
    {name: 'Suicidal Sentiment'},
    values,
  );
}

function convertToTensor(data) {
  return tf.tidy(() => {
    tf.util.shuffle(data);

    const inputs = data.map(d => d.Index)
    const labels = data.map(d => d.Suicidal_label);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });
}

async function trainModel(model, inputs, labels) {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 32;
  const epochs = 10;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

  model.add(tf.layers.dense({units: 1, useBias: true}));

  return model;
}

const model = createModel();
tfvis.show.modelSummary({name: 'Model Summary'}, model);

const csvFileInput = document.getElementById('csvFileInput');
csvFileInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if(file) {
    csvToJson(file, async(jsonData) => {
      test_data = jsonData;
      run(test_data);

      let tensorData = convertToTensor(test_data);
      const { inputs, labels } = tensorData;

      await trainModel(model, inputs, labels);
      console.log('finished training.');
      testModel(model, test_data, tensorData);
    });
  }
});
