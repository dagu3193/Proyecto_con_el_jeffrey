var mobilenet,
  classifier,
  video,
  derechaButton,
  entrenarButton,
  etiqueta = "-";

function preload() {
  video = createCapture(VIDEO);
}
function setup() {
  createCanvas(640, 480);
  if (video) {
    video.hide();
  }
  console.log(ml5);
  mobilenet = ml5.featureExtractor("MobileNet", modelLoaded);
  classifier = mobilenet.classification(video, videoReady);

  derechaButton = createButton("derecha");
  derechaButton.mousePressed(async () => {
    classifier.addImage("derecha");
  });
  izquierdaButton = createButton("izquierda");
  izquierdaButton.mousePressed(async () => {
    classifier.addImage("izquierda");
  });
  entrenarButton = createButton("entrenar");
  entrenarButton.mousePressed(async () => {
    classifier.train((loss) => {
      if (loss) {
        console.log(loss);
      } else {
        console.log("Entrenamiento finalizado");
        console.log(classifier.classify(obtenerResultado));
      }
    });
  });

  guardarButton = createButton("guardar");
  guardarButton.mousePressed(() => {
    descargar();
  });

  function descargar() {
    mobilenet.save((err, result) => {
      if (err) {
        window.alert("Failed to save model");
        console.error(err);
      }
    });
  }

  function obtenerResultado(error, results) {
    if (error) {
      console.error(error);
    } else {
      if (results[0].confidence < 0.1) {
        etiqueta = "----";
      } else {
        etiqueta = results[0].label;
      }
      classifier.classify(obtenerResultado);
    }
  }

  function videoReady() {
    console.log("El video está listo");
  }

  function modelLoaded() {
    console.log("El modelo ha sido cargado");
  }
}

function draw() {
  background(0);

  image(video, 0, 0, width, (width * video.height) / video.width);
  fill("black");
  rect(0, 0, width, 40);
  fill("white");
  textSize(20);
  textAlign(LEFT, TOP);
  text(etiqueta, 10, 10);
}
