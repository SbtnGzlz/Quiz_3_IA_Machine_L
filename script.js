let modelo = null;
let scalerMean = [51.96666666666667, 134.84761904761905, 224.21428571428572, 0.5, 0.4857142857142857];   // <-- Cambiar si usas otros valores
let scalerStd  = [15.844597496978894, 25.26423444219344, 45.398518810684585, 0.5, 0.4997958767010258];     // <-- Cambiar si usas otros valores

// CARGAR MODELO (guardamos la promesa)
async function cargarModelo() {
    try {
        modelo = await tf.loadLayersModel('model/nn/model.json');
        console.log("Modelo cargado OK");
        // opcional: habilitar botón predicción aquí si estaba deshabilitado
    } catch (err) {
        console.error("Error cargando el modelo:", err);
    }
}
cargarModelo();

function escalar(valores) {
    // valores: array de números [edad, sexo, peso, presion, colesterol]
    return valores.map((v,i) => (v - scalerMean[i]) / scalerStd[i]);
}

async function predecir() {
    if (!modelo) {
        alert("El modelo aún no está cargado. Espera unos segundos e intenta de nuevo.");
        console.error("Intento de predecir sin modelo cargado.");
        return;
    }

    // lectura de inputs
    let edad = parseFloat(document.getElementById("edad").value) || 0;
    let sexo = parseFloat(document.getElementById("sexo").value) || 0;
    let peso = parseFloat(document.getElementById("peso").value) || 0;
    let presion = parseFloat(document.getElementById("presion").value) || 0;
    let colesterol = parseFloat(document.getElementById("colesterol").value) || 0;

    // vector de entrada, sin doble corchete
    let entrada = [edad, sexo, peso, presion, colesterol];

    // aplica escala SI y SOLO SI tus datos en notebook fueron escalados
    let entradaEscalada = escalar(entrada);

    // aquí está la corrección: tensor2d con 1 fila y N columnas -> [entradaEscalada]
    const tensor = tf.tensor2d([entradaEscalada]);

    // predicción
    try {
        let pred = await modelo.predict(tensor).data();
        let prob = pred[0];
        let res = (prob >= 0.5) ? "✔ Sobrevive" : "✖ No Sobrevive";
        document.getElementById("resultado").innerText =
            `Resultado: ${res}  (prob: ${prob.toFixed(3)})`;
    } catch (err) {
        console.error("Error durante predict():", err);
        alert("Error al predecir (mira la consola para más detalles).");
    }

    // liberar tensor
    tensor.dispose();
}
