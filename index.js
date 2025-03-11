let classifier, video, faceMesh;
let derechaButton, izquierdaButton, entrenarButton, guardarButton;
let etiqueta = "-";
let faces = [];

function preload() {
  // Iniciamos la captura de video y lo ocultamos (ya que lo dibujaremos en el canvas)
  video = createCapture(VIDEO);
  video.hide();

  // Opciones para FaceMesh
  let options = { maxFaces: 1, refineLandmarks: false, flipHorizontal: false };
  // Inicializa FaceMesh con el video, las opciones y una función de callback
  faceMesh = ml5.faceMesh(options, modelLoadedFacemesh);
}

function setup() {
  createCanvas(640, 480);

  // Inicializa el clasificador usando ml5.imageClassifier para transfer learning con MobileNet
  classifier = ml5.imageClassifier("MobileNet", video, modelLoaded);

  // Botón para etiquetar la imagen como "derecha"
  derechaButton = createButton("derecha");
  derechaButton.mousePressed(() => {
    classifier.addImage("derecha");
  });

  // Botón para etiquetar la imagen como "izquierda"
  izquierdaButton = createButton("izquierda");
  izquierdaButton.mousePressed(() => {
    classifier.addImage("izquierda");
  });

  // Botón para entrenar el clasificador
  entrenarButton = createButton("entrenar");
  entrenarButton.mousePressed(() => {
    classifier.train((loss) => {
      if (loss) {
        console.log(loss);
      } else {
        console.log("Entrenamiento finalizado");
        classifier.classify(obtenerResultado);
      }
    });
  });

  // Botón para guardar el modelo entrenado
  guardarButton = createButton("guardar");
  guardarButton.mousePressed(() => {
    classifier.save();
  });
}

// Callback function for when faceMesh outputs data
function gotFaces(results) {
  // Save the output to the faces variable
  faces = results;
}

function modelLoadedFacemesh() {
  console.log("El modelo Facemesh ha sido cargado");

  // Start detecting faces from the webcam video
  faceMesh.detectStart(video, gotFaces);
}
function modelLoaded() {
  console.log("El modelo MobileNet ha sido cargado");
}

// Función asíncrona que llama recursivamente a faceMesh.predict()
function obtenerResultado(error, results) {
  if (error) {
    console.error(error);
  } else {
    if (results && results[0]) {
      if (results[0].confidence < 0.1) {
        etiqueta = "----";
      } else {
        etiqueta = results[0].label;
      }
    }
    // Vuelve a clasificar de forma continua
    classifier.classify(obtenerResultado);
  }
}

function draw() {
  background(0);
  // Dibuja el video en el canvas
  image(video, 0, 0, width, (width * video.height) / video.width);

  for (let i = 0; i < faces.length; i++) {
    let face = faces[i];

    const {
      box,
      lips,
      leftEye,
      leftEyebrow,
      rightEye,
      rightEyebrow,
      faceOval,
    } = face;

    push();
    {
      const { xMin, yMin, width, height } = box;
      stroke("red");
      fill(0, 0, 0, 0);
      rect(xMin, yMin, width, height);
    }
    pop();

    [
      { keypoints: face.keypoints, color: "rgba(180, 255,200,0.125)" },
      leftEye,
      leftEyebrow,
      rightEye,
      rightEyebrow,
      lips,
      faceOval,
    ].forEach((m) => {
      push();
      stroke(m.color ?? "lime");
      if(!m.color){
        strokeWeight(3);
      }
      fill(0, 0, 0, 0);
      {
        beginShape();
        const { keypoints } = m;
        for (let k = 0; k < keypoints.length; k++) {
          let keypoint = keypoints[k];
          vertex(keypoint.x, keypoint.y);
        }
        endShape(CLOSE);
      }
      pop();
    });

    /* for (let j = 0; j < face.scaledMesh.length; j++) {
      let keypoint = face.scaledMesh[j];
      fill(0, 255, 0);
      noStroke();
      circle(keypoint[0], keypoint[1], 5);
    } */
  }

  // Muestra la etiqueta en la parte superior
  fill("black");
  rect(0, 0, width, 40);
  fill("white");
  textSize(20);
  textAlign(LEFT, TOP);
  text(etiqueta, 10, 10);
}
