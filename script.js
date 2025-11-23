let modelo = null;
let scalerMean = [51.96666666666667, 134.84761904761905, 224.21428571428572, 0.5, 0.4857142857142857];   // <-- Cambiar si usas otros valores
let scalerStd  = [15.844597496978894, 25.26423444219344, 45.398518810684585, 0.5, 0.4997958767010258];     // <-- Cambiar si usas otros valores

// CARGAR MODELO de forma segura al cargar la página
async function cargarModelo() {
  try {
    console.log("Cargando TF.js model desde: model/nn/model.json ...");
    modelo = await tf.loadLayersModel('model/nn/model.json');
    console.log("Modelo cargado OK");
    // opcional: muestra resumen y habilita botón
    try { modelo.summary(); } catch(e) {}
    const btn = document.getElementById('predictBtn');
    if (btn) btn.disabled = false;
  } catch (err) {
    console.error("Error cargando modelo:", err);
    alert("Error cargando el modelo. Mira la consola para más detalles.");
  }
}

// Aseguramos que el cargado pase hasta que el documento esté listo
window.addEventListener('load', cargarModelo);

// Función de escala (asegúrate de usar valores numéricos)
function escalar(valores) {
  return valores.map((v, i) => {
    const num = Number(v);
    if (!isFinite(num)) return NaN;
    return (num - Number(scalerMean[i])) / Number(scalerStd[i]);
  });
}

async function predecir() {
  if (!modelo) {
    alert("El modelo aún no está cargado — espera unos segundos y prueba de nuevo.");
    console.error("Intento de predecir sin modelo cargado.");
    return;
  }

  // Leer inputs (asegúrate que tus inputs tengan estos ids: edad, sexo, peso, presion, colesterol)
  const ids = ['edad','sexo','peso','presion','colesterol'];
  const raw = ids.map(id => {
    const el = document.getElementById(id);
    const v = el ? el.value : "";
    return v;
  });

  console.log("Entrada raw:", raw);

  // Convertir a número y validar
  const nums = raw.map(v => {
    const n = Number(v);
    return isFinite(n) ? n : NaN;
  });

  if (nums.some(x => Number.isNaN(x))) {
    alert("Por favor completa todos los campos con valores numéricos válidos.");
    console.error("Inputs inválidos:", nums);
    return;
  }

  // Escalar (si tienes valores reales); si no quieres escalar temporalmente, reemplaza `entradaEscalada = nums`
  const entradaEscalada = escalar(nums);
  console.log("Entrada escalada:", entradaEscalada);

  if (entradaEscalada.some(x => Number.isNaN(x))) {
    alert("Error: hay NaN en la entrada escalada. Verifica scalerMean/scalerStd y los inputs.");
    console.error("NaN en entrada escalada:", entradaEscalada);
    return;
  }

  // Crear tensor 2D: 1 fila, N columnas
  const tensor = tf.tensor2d([entradaEscalada]); // forma: [1, 5]
  console.log("Tensor shape:", tensor.shape);

  try {
    const out = modelo.predict(tensor);
    const data = await out.data();
    const prob = data[0];
    const res = (prob >= 0.5) ? "✔ Sobrevive" : "✖ No Sobrevive";
    document.getElementById("resultado").innerText =
      `Resultado: ${res}  (prob: ${prob.toFixed(3)})`;
  } catch (err) {
    console.error("Error durante predict():", err);
    alert("Error al predecir. Revisa la consola.");
  } finally {
    tensor.dispose();
  }
}