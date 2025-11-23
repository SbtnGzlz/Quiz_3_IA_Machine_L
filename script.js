let modelo;
let scalerMean = [50, 0.5, 70, 120, 180];   // <-- Cambiar si usas otros valores
let scalerStd  = [10, 0.5, 15, 15, 30];     // <-- Cambiar si usas otros valores

// CARGAR MODELO
async function cargarModelo() {
    modelo = await tf.loadLayersModel('model/nn/model.json');
    console.log("Modelo cargado");
}
cargarModelo();

function escalar(valores) {
    return valores.map((v, i) => (v - scalerMean[i]) / scalerStd[i]);
}

async function predecir() {
    let edad = parseFloat(document.getElementById("edad").value);
    let sexo = parseFloat(document.getElementById("sexo").value);
    let peso = parseFloat(document.getElementById("peso").value);
    let presion = parseFloat(document.getElementById("presion").value);
    let colesterol = parseFloat(document.getElementById("colesterol").value);

    let entrada = [edad, sexo, peso, presion, colesterol];
    let entradaEscalada = escalar(entrada);

    const tensor = tf.tensor2d([entradaEscalada]);

    let pred = await modelo.predict(tensor).data();
    let prob = pred[0];

    let res = (prob >= 0.5) ? "✔ Sobrevive" : "✖ No Sobrevive";

    document.getElementById("resultado").innerText =
        `Resultado: ${res}  (prob: ${prob.toFixed(3)})`;
}
